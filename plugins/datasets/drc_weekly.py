"""Weekly climate and vegetation datasets used by the DRC DHIS2 pipeline.

The plugins in this module deliberately expose ISO-week periods (``YYYY-Www``)
and stamp every output on the Monday starting that week. Daily climate sources
are reduced over seven complete days. MODIS 16-day composites are converted to
weekly observations with an overlap-weighted mean; this improves period alignment
without pretending that MODIS has a native seven-day acquisition cycle.
"""

from __future__ import annotations

import asyncio
import calendar
import netrc
import os
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Literal, cast

import numpy as np
import xarray as xr
import yaml
from ecmwf.datastores import Client as _CdsClient

from open_climate_service.plugins.datasets.chirps3 import CHIRPS3DailyPlugin
from open_climate_service.plugins.datasets.era5_land import (
    ERA5LandPrecipDailyPlugin,
    ERA5LandTempDailyPlugin,
    _cds_end_datetime,
    _collapse_expver,
    _daily_availability_cutoff,
    _drop_auxiliary_variables,
    _hourly_availability_cutoff,
)
from open_climate_service.shared.time import normalize_period_string, parse_weekly_period_string
from open_climate_service.streaming import BaseDatasetPlugin, daily_period_ids
from open_climate_service.transforms.unit_conversion import kelvin_to_celsius

_Reducer = Literal["sum", "mean", "max", "min"]
_ERA5_HEAT_ARCO_URL = (
    "https://arco.datastores.ecmwf.int/cadl-arco-time-004/"
    "arco/derived_utci_historical/all/timeChunked.zarr"
)
_MODIS_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_MODIS_COLLECTION = "modis-13A1-061"
_MODIS_BEGIN = date(2000, 2, 18)
_CDS_API_URL = "https://cds.climate.copernicus.eu/api"


def _week_monday(value: str) -> date:
    """Return the Monday of an ISO week or ISO date/datetime value."""
    canonical = normalize_period_string(str(value), "weekly")
    return parse_weekly_period_string(canonical).date()


def _week_id(monday: date) -> str:
    """Return the canonical ISO-week identifier for a Monday."""
    iso = monday.isocalendar()
    return f"{iso.year:04d}-W{iso.week:02d}"


def _week_days(period_id: str) -> list[date]:
    """Return the seven dates belonging to a weekly period."""
    monday = _week_monday(period_id)
    return [monday + timedelta(days=offset) for offset in range(7)]


def _complete_week_ids(start: str, end: str, cutoff: date) -> list[str]:
    """Enumerate complete ISO weeks inside a requested inclusive range."""
    current = _week_monday(start)
    requested_last = _week_monday(end)
    last_complete = cutoff - timedelta(days=cutoff.weekday())
    if last_complete + timedelta(days=6) > cutoff:
        last_complete -= timedelta(days=7)
    limit = min(requested_last, last_complete)
    result: list[str] = []
    while current <= limit:
        result.append(_week_id(current))
        current += timedelta(days=7)
    return result


class _CachedLatestMixin:
    """Cache the EDH availability probe for the lifetime of an ingest."""

    _weekly_latest: str | None = None

    def _latest_available(self) -> str:
        if self._weekly_latest is None:
            self._weekly_latest = super()._latest_available()  # type: ignore[misc]
        return self._weekly_latest


class _CachedERA5LandTempDaily(_CachedLatestMixin, ERA5LandTempDailyPlugin):
    """ERA5-Land daily temperature with one EDH availability lookup."""


class _CachedERA5LandPrecipDaily(_CachedLatestMixin, ERA5LandPrecipDailyPlugin):
    """ERA5-Land daily precipitation with one EDH availability lookup."""


def _edh_credentials_available() -> bool:
    """Return whether the optional DestinE Earth Data Hub credentials exist."""
    if os.getenv("EDH_API_KEY", "").strip():
        return True
    try:
        return netrc.netrc().authenticators("api.earthdatahub.destine.eu") is not None
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False


class _CDSOnlyERA5LandPrecipDaily(ERA5LandPrecipDailyPlugin):
    """ERA5-Land daily precipitation using CDS when EDH is not configured."""

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = await asyncio.to_thread(_hourly_availability_cutoff)
        return daily_period_ids(start, end, cutoff=cutoff.date())

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        _configure_ecmwf_datastores()
        return _drop_auxiliary_variables(self._fetch_cds_daily_sync(period_id, bbox), "tp")


class _WeeklyFromDailyPlugin(BaseDatasetPlugin):
    """Reduce seven complete daily source periods into one ISO-week period."""

    max_concurrency = 1
    commit_batch_size = 4

    def __init__(
        self,
        source: BaseDatasetPlugin,
        source_variable: str,
        output_variable: str,
        reducer: _Reducer,
    ) -> None:
        self.source = source
        self.source_variable = source_variable
        self.output_variable = output_variable
        self.reducer = reducer

    async def periods(self, start: str, end: str) -> list[str]:
        first = _week_monday(start)
        last = _week_monday(end)
        available = set(
            await self.source.periods(
                first.isoformat(),
                (last + timedelta(days=6)).isoformat(),
            )
        )
        result: list[str] = []
        current = first
        while current <= last:
            if all(day.isoformat() in available for day in _week_days(_week_id(current))):
                result.append(_week_id(current))
            current += timedelta(days=7)
        return result

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        arrays: list[xr.DataArray] = []
        for day in _week_days(period_id):
            daily = self.source.fetch_period(day.isoformat(), bbox)
            if not isinstance(daily, xr.Dataset):
                raise TypeError("Daily source returned an awaitable to a synchronous weekly plugin")
            try:
                arrays.append(daily[self.source_variable].load())
            finally:
                daily.close()

        combined = xr.concat(arrays, dim="t")
        if self.reducer == "sum":
            reduced = combined.sum("t", skipna=True, min_count=1)
        elif self.reducer == "mean":
            reduced = combined.mean("t", skipna=True)
        elif self.reducer == "max":
            reduced = combined.max("t", skipna=True)
        else:
            reduced = combined.min("t", skipna=True)

        monday = np.datetime64(_week_monday(period_id).isoformat(), "D").astype("datetime64[ns]")
        return reduced.astype("float32").expand_dims(t=[monday]).to_dataset(name=self.output_variable)

    def close(self) -> None:
        """Release any cached remote datasets owned by the daily source."""
        close = getattr(self.source, "close", None)
        if callable(close):
            close()


class CHIRPS3PrecipitationWeeklyPlugin(_WeeklyFromDailyPlugin):
    """CHIRPS3 weekly precipitation total in millimetres."""

    def __init__(self, **_: Any) -> None:
        """Build the weekly reducer around the CHIRPS3 daily source."""
        super().__init__(CHIRPS3DailyPlugin(), "precip", "precip", "sum")


class ERA5LandPrecipitationWeeklyPlugin(_WeeklyFromDailyPlugin):
    """ERA5-Land weekly precipitation total in millimetres."""

    def __init__(self, **_: Any) -> None:
        """Build the weekly reducer around ERA5-Land daily precipitation."""
        source = _CachedERA5LandPrecipDaily() if _edh_credentials_available() else _CDSOnlyERA5LandPrecipDaily()
        super().__init__(source, "tp", "tp", "sum")


class ERA5LandTemperatureWeeklyPlugin(_WeeklyFromDailyPlugin):
    """ERA5-Land weekly mean 2-metre air temperature."""

    def __init__(self, **_: Any) -> None:
        """Build the weekly reducer around ERA5-Land daily mean temperature."""
        source: BaseDatasetPlugin
        if _edh_credentials_available():
            source = _CachedERA5LandTempDaily(variable="t2m")
        else:
            source = _ERA5LandDailyStatisticPlugin("t2m", "mean")
        super().__init__(source, "t2m", "t2m", "mean")


class _ERA5LandDailyStatisticPlugin(BaseDatasetPlugin):
    """Download one month of an ERA5-Land daily temperature statistic."""

    max_concurrency = 1
    commit_batch_size = 31
    _variables = {"t2m": "2m_temperature", "d2m": "2m_dewpoint_temperature"}
    _statistics = {"mean": "daily_mean", "max": "daily_maximum", "min": "daily_minimum"}

    def __init__(self, variable: str, statistic: Literal["mean", "max", "min"]) -> None:
        if variable not in self._variables:
            raise ValueError(f"Unsupported ERA5-Land variable: {variable}")
        self.variable = variable
        self.statistic = statistic
        self._cache_lock = Lock()
        self._cached_month: tuple[int, int] | None = None
        self._cached_bbox: tuple[float, float, float, float] | None = None
        self._cached_ds: xr.Dataset | None = None

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = await asyncio.to_thread(_daily_availability_cutoff)
        return daily_period_ids(start, end, cutoff=cutoff)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        day = date.fromisoformat(period_id)
        bbox_tuple = cast(tuple[float, float, float, float], tuple(map(float, bbox)))
        with self._cache_lock:
            if self._cached_month != (day.year, day.month) or self._cached_bbox != bbox_tuple:
                self._cached_ds = self._fetch_month(day.year, day.month, bbox_tuple)
                self._cached_month = (day.year, day.month)
                self._cached_bbox = bbox_tuple
            monthly = self._cached_ds
        assert monthly is not None
        timestamp = np.datetime64(period_id, "D").astype("datetime64[ns]")
        return _drop_auxiliary_variables(monthly.sel(t=slice(timestamp, timestamp)), self.variable)

    def _fetch_month(self, year: int, month: int, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        xmin, ymin, xmax, ymax = bbox
        last_day = calendar.monthrange(year, month)[1]
        cutoff = _daily_availability_cutoff()
        if (year, month) == (cutoff.year, cutoff.month):
            last_day = min(last_day, cutoff.day)
        params = {
            "variable": [self._variables[self.variable]],
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{day:02d}" for day in range(1, last_day + 1)],
            "daily_statistic": self._statistics[self.statistic],
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": [ymax, xmin, ymin, xmax],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "era5land_daily_statistic.nc"
            _cds_client().submit("derived-era5-land-daily-statistics", params).download(str(target))
            dataset = xr.open_dataset(target, engine="netcdf4").load()
        dataset = _collapse_expver(dataset[[self.variable]])
        time_dim = "valid_time" if "valid_time" in dataset.dims else "time"
        dataset = dataset.rename({"longitude": "x", "latitude": "y", time_dim: "t"})
        return kelvin_to_celsius(dataset, {"variable": self.variable})


class ERA5LandTemperatureMaxWeeklyPlugin(_WeeklyFromDailyPlugin):
    """ERA5-Land maximum 2-metre air temperature observed during a week."""

    def __init__(self, **_: Any) -> None:
        """Build the weekly reducer for maximum temperature."""
        source = _ERA5LandDailyStatisticPlugin("t2m", "max")
        super().__init__(source, "t2m", "t2m_max", "max")


class ERA5LandTemperatureMinWeeklyPlugin(_WeeklyFromDailyPlugin):
    """ERA5-Land minimum 2-metre air temperature observed during a week."""

    def __init__(self, **_: Any) -> None:
        """Build the weekly reducer for minimum temperature."""
        source = _ERA5LandDailyStatisticPlugin("t2m", "min")
        super().__init__(source, "t2m", "t2m_min", "min")


class ERA5LandDewpointTemperatureWeeklyPlugin(_WeeklyFromDailyPlugin):
    """ERA5-Land weekly mean 2-metre dew-point temperature."""

    def __init__(self, **_: Any) -> None:
        """Build the weekly reducer for mean dew-point temperature."""
        source = _ERA5LandDailyStatisticPlugin("d2m", "mean")
        super().__init__(source, "d2m", "d2m", "mean")


class _ERA5LandDailyRelativeHumidityPlugin(BaseDatasetPlugin):
    """Derive daily relative humidity from mean air and dew-point temperature."""

    max_concurrency = 1

    def __init__(self) -> None:
        self.temperature = _ERA5LandDailyStatisticPlugin("t2m", "mean")
        self.dewpoint = _ERA5LandDailyStatisticPlugin("d2m", "mean")

    async def periods(self, start: str, end: str) -> list[str]:
        temperature, dewpoint = await asyncio.gather(
            self.temperature.periods(start, end),
            self.dewpoint.periods(start, end),
        )
        return sorted(set(temperature) & set(dewpoint))

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        temperature = self.temperature.fetch_period(period_id, bbox)
        dewpoint = self.dewpoint.fetch_period(period_id, bbox)
        try:
            air = temperature["t2m"].load()
            dew = dewpoint["d2m"].load()
        finally:
            temperature.close()
            dewpoint.close()
        exponent = (17.625 * dew / (243.04 + dew)) - (17.625 * air / (243.04 + air))
        humidity = cast(xr.DataArray, 100.0 * xr.apply_ufunc(np.exp, exponent))
        humidity = humidity.clip(0.0, 100.0).astype("float32")
        return humidity.to_dataset(name="hurs")


class ERA5LandRelativeHumidityWeeklyPlugin(_WeeklyFromDailyPlugin):
    """ERA5-Land weekly mean relative humidity derived from temperature and dew point."""

    def __init__(self, **_: Any) -> None:
        """Build the weekly reducer for derived daily relative humidity."""
        super().__init__(_ERA5LandDailyRelativeHumidityPlugin(), "hurs", "hurs", "mean")


def _read_yaml_key(path: Path) -> str:
    """Read a ``key`` value from a CDS-style YAML file."""
    if not path.exists():
        return ""
    parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(parsed, dict):
        return ""
    return str(parsed.get("key", "")).strip()


def _cds_api_key() -> str:
    """Resolve one CDS token for both CDS client implementations."""
    for variable in ("ECMWF_DATASTORES_KEY", "CDSAPI_KEY"):
        token = os.getenv(variable, "").strip()
        if token:
            return token
    for path in (Path.home() / ".ecmwfdatastoresrc", Path.home() / ".cdsapirc"):
        token = _read_yaml_key(path)
        if token:
            return token
    raise RuntimeError(
        "Copernicus CDS credentials are missing. Set CDSAPI_KEY in .env "
        "(recommended), set ECMWF_DATASTORES_KEY, or create ~/.cdsapirc."
    )


def _configure_ecmwf_datastores() -> tuple[str, str]:
    """Bridge the project's CDS variables to ecmwf-datastores-client."""
    url = (
        os.getenv("ECMWF_DATASTORES_URL", "").strip()
        or os.getenv("CDSAPI_URL", "").strip()
        or _CDS_API_URL
    )
    token = _cds_api_key()
    if not os.getenv("ECMWF_DATASTORES_URL", "").strip():
        os.environ["ECMWF_DATASTORES_URL"] = url
    if not os.getenv("ECMWF_DATASTORES_KEY", "").strip():
        os.environ["ECMWF_DATASTORES_KEY"] = token
    return url, token


def _cds_client() -> _CdsClient:
    """Create a CDS client without depending on an undocumented rc filename."""
    url, token = _configure_ecmwf_datastores()
    return _CdsClient(url=url, key=token)


def _coordinate_name(dataset: xr.Dataset | xr.DataArray, *candidates: str) -> str:
    """Resolve a coordinate/dimension name from common source variants."""
    for candidate in candidates:
        if candidate in dataset.coords or candidate in dataset.dims:
            return candidate
    raise ValueError(f"None of the expected coordinates {candidates!r} exist")


def _ordered_slice(coordinate: xr.DataArray, minimum: float, maximum: float) -> slice:
    """Build a slice that respects ascending or descending coordinates."""
    first = float(coordinate.isel({coordinate.dims[0]: 0}))
    last = float(coordinate.isel({coordinate.dims[0]: -1}))
    return slice(maximum, minimum) if first > last else slice(minimum, maximum)


class ERA5HeatWeeklyPlugin(BaseDatasetPlugin):
    """Weekly mean or extreme UTCI from the authenticated ERA5-HEAT ARCO store."""

    max_concurrency = 1
    commit_batch_size = 4

    def __init__(self, statistic: Literal["mean", "max", "min"] = "mean", **_: Any) -> None:
        """Configure the temporal UTCI statistic to calculate."""
        if statistic not in {"mean", "max", "min"}:
            raise ValueError("ERA5HeatWeeklyPlugin statistic must be mean, max or min")
        self.statistic = statistic
        self.output_variable = "utci" if statistic == "mean" else f"utci_{statistic}"
        self._source_lock = Lock()
        self._source: xr.Dataset | None = None

    async def periods(self, start: str, end: str) -> list[str]:
        """Return complete ISO weeks available from ERA5-HEAT."""
        cutoff = await asyncio.to_thread(_cds_end_datetime, "derived-utci-historical")
        return _complete_week_ids(start, end, cutoff.date())

    def _open_source(self) -> xr.Dataset:
        with self._source_lock:
            if self._source is None:
                self._source = xr.open_zarr(
                    _ERA5_HEAT_ARCO_URL,
                    storage_options={"headers": {"Authorization": f"Bearer {_cds_api_key()}"}},
                    chunks="auto",
                    consolidated=None,
                )
            return self._source

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """Read and reduce the UTCI hours intersecting one ISO week."""
        source = self._open_source()
        variable = source["utci"]
        time_name = _coordinate_name(variable, "valid_time", "time", "t")
        latitude_name = _coordinate_name(variable, "latitude", "lat", "y")
        longitude_name = _coordinate_name(variable, "longitude", "lon", "x")
        monday = _week_monday(period_id)
        sunday = monday + timedelta(days=6)
        xmin, ymin, xmax, ymax = map(float, bbox)
        selection = variable.sel(
            {
                time_name: slice(
                    np.datetime64(monday),
                    np.datetime64(sunday) + np.timedelta64(1, "D") - np.timedelta64(1, "ns"),
                ),
                latitude_name: _ordered_slice(variable[latitude_name], ymin, ymax),
                longitude_name: _ordered_slice(variable[longitude_name], xmin, xmax),
            }
        ).load()
        if selection.sizes.get(time_name, 0) == 0:
            raise RuntimeError(f"ERA5-HEAT returned no UTCI values for {period_id}")
        if self.statistic == "mean":
            reduced = selection.mean(time_name, skipna=True)
        elif self.statistic == "max":
            reduced = selection.max(time_name, skipna=True)
        else:
            reduced = selection.min(time_name, skipna=True)
        units = str(selection.attrs.get("units", "K")).lower()
        if units in {"k", "kelvin"}:
            reduced = reduced - 273.15
        reduced = reduced.rename({latitude_name: "y", longitude_name: "x"}).astype("float32")
        timestamp = np.datetime64(monday.isoformat(), "D").astype("datetime64[ns]")
        return reduced.expand_dims(t=[timestamp]).to_dataset(name=self.output_variable)

    def close(self) -> None:
        """Close the lazily opened ARCO source."""
        with self._source_lock:
            if self._source is not None:
                self._source.close()
            self._source = None


def _item_interval(item: Any) -> tuple[date, date]:
    """Return a STAC item's inclusive composite interval."""
    metadata = item.common_metadata
    start = metadata.start_datetime or item.datetime
    end = metadata.end_datetime or item.datetime
    if start is None or end is None:
        raise ValueError(f"MODIS STAC item {item.id!r} has no temporal interval")
    return start.date(), end.date()


def _is_terra_mod13a1(item: Any) -> bool:
    """Recognise Terra items across STAC metadata variants."""
    platform = str(item.properties.get("platform", "")).strip().lower()
    item_id = str(getattr(item, "id", "")).upper()
    return platform == "terra" or item_id.startswith("MOD13A1")


def _overlap_days(first_start: date, first_end: date, second_start: date, second_end: date) -> int:
    """Count inclusive overlapping days between two intervals."""
    return max(0, (min(first_end, second_end) - max(first_start, second_start)).days + 1)


class MODISVegetationWeeklyPlugin(BaseDatasetPlugin):
    """Weekly NDVI or EVI from Terra MOD13A1 composites on a 0.01-degree grid."""

    max_concurrency = 1
    commit_batch_size = 4
    _assets = {
        "ndvi": "500m_16_days_NDVI",
        "evi": "500m_16_days_EVI",
    }
    _quality_asset = "500m_16_days_pixel_reliability"

    def __init__(self, index: Literal["ndvi", "evi"], resolution_degrees: float = 0.01, **_: Any) -> None:
        """Configure the vegetation index and output grid resolution."""
        if index not in self._assets:
            raise ValueError("MODIS vegetation index must be ndvi or evi")
        self.index = index
        self.resolution_degrees = float(resolution_degrees)
        self._client: Any | None = None

    def _catalog(self) -> Any:
        if self._client is None:
            import planetary_computer  # type: ignore[import-untyped]
            import pystac_client

            self._client = pystac_client.Client.open(_MODIS_STAC_URL, modifier=planetary_computer.sign_inplace)
        return self._client

    async def periods(self, start: str, end: str) -> list[str]:
        """Return complete ISO weeks covered by Terra MOD13A1 composites."""
        requested_end = _week_monday(end) + timedelta(days=6)
        if requested_end < _MODIS_BEGIN:
            return []
        cutoff = await asyncio.to_thread(self._latest_available, requested_end)
        first = max(_week_monday(start), _week_monday(_MODIS_BEGIN.isoformat()))
        return _complete_week_ids(first.isoformat(), end, cutoff)

    def _latest_available(self, requested_end: date) -> date:
        from open_climate_service.config import get_config

        extent = get_config().get("extent", {})
        bbox = extent.get("bbox", [12.0, -13.5, 31.5, 5.4])
        # A pipeline may ask for a future end period. Searching STAC beyond today
        # can return an empty page even though historical composites are available.
        effective_end = min(requested_end, datetime.now(UTC).date())
        for lookback_days in (64, 366, (effective_end - _MODIS_BEGIN).days):
            search_start = max(_MODIS_BEGIN, effective_end - timedelta(days=max(0, lookback_days)))
            items = list(
                self._catalog()
                .search(
                    collections=[_MODIS_COLLECTION],
                    bbox=bbox,
                    datetime=(
                        f"{search_start.isoformat()}T00:00:00Z/"
                        f"{effective_end.isoformat()}T23:59:59Z"
                    ),
                )
                .items()
            )
            terra = [item for item in items if _is_terra_mod13a1(item)]
            if terra:
                return min(effective_end, max(_item_interval(item)[1] for item in terra))
        raise RuntimeError(
            "No Terra MOD13A1 composite was found for the configured extent up to "
            f"{effective_end.isoformat()}; verify Planetary Computer connectivity and the extent bbox"
        )

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """Load quality-screened composites and calculate one weekly index."""
        from odc.stac import load

        monday = _week_monday(period_id)
        sunday = monday + timedelta(days=6)
        items = list(
            self._catalog()
            .search(
                collections=[_MODIS_COLLECTION],
                bbox=bbox,
                datetime=f"{monday.isoformat()}T00:00:00Z/{sunday.isoformat()}T23:59:59Z",
            )
            .items()
        )
        terra = [item for item in items if _is_terra_mod13a1(item)]
        if not terra:
            raise RuntimeError(f"No Terra MOD13A1 composite overlaps {period_id}")

        asset = self._assets[self.index]
        cube = load(
            terra,
            bands=[asset, self._quality_asset],
            bbox=cast(tuple[float, float, float, float], tuple(bbox)),
            crs="EPSG:4326",
            resolution=self.resolution_degrees,
            groupby="solar_day",
            resampling="nearest",
            # Planetary Computer declares pixel reliability as int8 while
            # source tiles expose nodata as either -1 or 255. int16 safely
            # represents both and avoids odc-stac's nodata cast overflow.
            dtype=cast(Any, {asset: "int16", self._quality_asset: "int16"}),
        ).load()
        raw = cube[asset].astype("float32")
        quality = cube[self._quality_asset]
        values = (raw * np.float32(0.0001)).where(
            ((quality == 0) | (quality == 1)) & (raw >= -2000) & (raw <= 10000)
        )

        intervals = {_item_interval(item)[0]: _item_interval(item) for item in terra}
        weights: list[float] = []
        for timestamp in cube.time.values:
            acquisition = np.datetime64(timestamp, "D").astype(object)
            if not isinstance(acquisition, date):
                raise TypeError("MODIS acquisition timestamp did not convert to date")
            interval = intervals.get(acquisition, (acquisition, acquisition + timedelta(days=15)))
            weights.append(float(_overlap_days(interval[0], interval[1], monday, sunday)))
        weight = xr.DataArray(weights, coords={"time": cube.time}, dims=("time",))
        denominator = weight.where(values.notnull()).sum("time", skipna=True)
        weekly = ((values * weight).sum("time", skipna=True) / denominator).where(denominator > 0)
        rename: dict[str, str] = {}
        if "latitude" in weekly.dims:
            rename["latitude"] = "y"
        if "longitude" in weekly.dims:
            rename["longitude"] = "x"
        weekly = weekly.rename(rename).astype("float32")
        if "y" not in weekly.dims or "x" not in weekly.dims:
            raise ValueError(f"Unexpected MODIS spatial dimensions: {weekly.dims!r}")
        timestamp = np.datetime64(monday.isoformat(), "D").astype("datetime64[ns]")
        return weekly.expand_dims(t=[timestamp]).to_dataset(name=self.index)
