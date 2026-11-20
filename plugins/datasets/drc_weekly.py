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
import json
import logging
import netrc
import os
import shutil
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Literal, cast
from urllib.request import Request, urlopen

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
    "https://arco.datastores.ecmwf.int/cadl-arco-time-004/arco/derived_utci_historical/all/timeChunked.zarr"
)
_MODIS_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_MODIS_COLLECTION = "modis-13A1-061"
_MODIS_BEGIN = date(2000, 2, 18)
_CDS_API_URL = "https://cds.climate.copernicus.eu/api"
_EARTH_ENGINE_ERA5_LAND_DAILY = "ECMWF/ERA5_LAND/DAILY_AGGR"
_EARTH_ENGINE_ERA5_LAND_CRS = "EPSG:4326"
_EARTH_ENGINE_ERA5_LAND_TRANSFORM = [0.1, 0.0, -180.05, 0.0, -0.1, 90.05]
_EARTH_ENGINE_INITIALIZATION_LOCK = Lock()
_earth_engine_initialized_project: str | None = None

logger = logging.getLogger(__name__)


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


class _DelegatingWeeklyPlugin(BaseDatasetPlugin):
    """Expose one configured weekly provider through the streaming protocol."""

    max_concurrency = 1
    commit_batch_size = 4

    def __init__(self, delegate: BaseDatasetPlugin) -> None:
        self._delegate = delegate

    async def periods(self, start: str, end: str) -> list[str]:
        return await self._delegate.periods(start, end)

    def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        result = self._delegate.fetch_period(period_id, bbox, **params)
        if not isinstance(result, xr.Dataset):
            raise TypeError("Weekly provider returned an awaitable to a synchronous plugin")
        return result

    def close(self) -> None:
        close = getattr(self._delegate, "close", None)
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


def _era5_daily_statistics_provider() -> Literal["cds", "earth_engine"]:
    """Return the explicitly configured ERA5-Land daily statistics provider."""
    configured = os.getenv("OCS_ERA5_DAILY_PROVIDER", "cds").strip().lower().replace("-", "_")
    aliases = {"gee": "earth_engine", "earthengine": "earth_engine"}
    provider = aliases.get(configured, configured)
    if provider not in {"cds", "earth_engine"}:
        raise ValueError("OCS_ERA5_DAILY_PROVIDER must be either 'cds' or 'earth_engine'")
    return cast(Literal["cds", "earth_engine"], provider)


def _positive_float_environment(name: str, default: float) -> float:
    """Read a positive floating-point setting from the environment."""
    raw = os.getenv(name, "").strip()
    value = float(raw) if raw else default
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _earth_engine_client() -> Any:
    """Initialize and return the Earth Engine module for unattended use."""
    global _earth_engine_initialized_project

    try:
        import ee
    except ImportError as error:
        raise RuntimeError(
            "Earth Engine support is not installed. Install the 'earth-engine' optional dependency."
        ) from error

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    project = os.getenv("EARTH_ENGINE_PROJECT", "").strip() or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    service_account = os.getenv("EARTH_ENGINE_SERVICE_ACCOUNT", "").strip()
    credentials: Any | None = None

    if credentials_path:
        path = Path(credentials_path).expanduser()
        if not path.is_file():
            raise RuntimeError(f"Google service-account credentials do not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("Google service-account credentials must contain a JSON object")
        service_account = service_account or str(payload.get("client_email", "")).strip()
        project = project or str(payload.get("project_id", "")).strip()
        if not service_account:
            raise RuntimeError("Google service-account credentials do not contain client_email")
        credentials = ee.ServiceAccountCredentials(service_account, str(path))

    if not project:
        raise RuntimeError(
            "Earth Engine requires EARTH_ENGINE_PROJECT or GOOGLE_CLOUD_PROJECT to identify the Cloud project"
        )

    with _EARTH_ENGINE_INITIALIZATION_LOCK:
        if _earth_engine_initialized_project != project:
            if credentials is None:
                ee.Initialize(project=project)
            else:
                ee.Initialize(credentials, project=project)
            _earth_engine_initialized_project = project
            logger.info("Earth Engine initialized for ERA5-Land daily statistics | project=%s", project)
    return ee


class _EarthEngineERA5LandWeeklyPlugin(BaseDatasetPlugin):
    """Aggregate one ERA5-Land ISO week on Earth Engine and download one GeoTIFF."""

    max_concurrency = 1
    commit_batch_size = 4

    def __init__(
        self,
        source_band: str | None,
        output_variable: str,
        reducer: Literal["mean", "max", "min"],
        *,
        convert_kelvin: bool = False,
        relative_humidity: bool = False,
    ) -> None:
        if source_band is None and not relative_humidity:
            raise ValueError("An Earth Engine source band is required unless relative humidity is requested")
        self.source_band = source_band
        self.output_variable = output_variable
        self.reducer = reducer
        self.convert_kelvin = convert_kelvin
        self.relative_humidity = relative_humidity
        self._latest_date: date | None = None

    def _collection(self) -> Any:
        ee = _earth_engine_client()
        return ee.ImageCollection(_EARTH_ENGINE_ERA5_LAND_DAILY)

    def _latest_available(self) -> date:
        if self._latest_date is None:
            latest = self._collection().aggregate_max("system:time_start").getInfo()
            if latest is None:
                raise RuntimeError(f"Earth Engine dataset {_EARTH_ENGINE_ERA5_LAND_DAILY} is empty")
            self._latest_date = datetime.fromtimestamp(float(latest) / 1000.0, tz=UTC).date()
        return self._latest_date

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = await asyncio.to_thread(self._latest_available)
        return _complete_week_ids(start, end, cutoff)

    def _relative_humidity_image(self, collection: Any) -> Any:
        ee = _earth_engine_client()

        def daily_relative_humidity(source: Any) -> Any:
            image = ee.Image(source)
            air = image.select("temperature_2m").subtract(273.15)
            dewpoint = image.select("dewpoint_temperature_2m").subtract(273.15)
            air_term = air.multiply(17.625).divide(air.add(243.04))
            dewpoint_term = dewpoint.multiply(17.625).divide(dewpoint.add(243.04))
            return dewpoint_term.subtract(air_term).exp().multiply(100.0).clamp(0.0, 100.0).rename("hurs")

        return collection.map(daily_relative_humidity).mean()

    def _weekly_image(self, period_id: str) -> Any:
        monday = _week_monday(period_id)
        end_exclusive = monday + timedelta(days=7)
        collection = self._collection().filterDate(monday.isoformat(), end_exclusive.isoformat())
        if self.relative_humidity:
            image = self._relative_humidity_image(collection)
        else:
            assert self.source_band is not None
            selected = collection.select(self.source_band)
            if self.reducer == "mean":
                image = selected.mean()
            elif self.reducer == "max":
                image = selected.max()
            else:
                image = selected.min()
            if self.convert_kelvin:
                image = image.subtract(273.15)
        return image.rename(self.output_variable)

    def _download_image(self, image: Any, bbox: list[float]) -> xr.DataArray:
        if len(bbox) != 4:
            raise ValueError("Earth Engine downloads require a four-value bounding box")
        xmin, ymin, xmax, ymax = map(float, bbox)
        ee = _earth_engine_client()
        region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], proj="EPSG:4326", geodesic=False)
        timeout = _positive_float_environment("OCS_EARTH_ENGINE_DOWNLOAD_TIMEOUT_SECONDS", 300.0)
        url = image.getDownloadURL(
            {
                "region": region,
                "crs": _EARTH_ENGINE_ERA5_LAND_CRS,
                "crs_transform": _EARTH_ENGINE_ERA5_LAND_TRANSFORM,
                "format": "GEO_TIFF",
            }
        )
        request = Request(str(url), headers={"User-Agent": "open-climate-service/0.1"})
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "era5land_weekly.tif"
            with urlopen(request, timeout=timeout) as response, target.open("wb") as destination:
                shutil.copyfileobj(response, destination)

            import rioxarray

            source = cast(xr.DataArray, rioxarray.open_rasterio(target, masked=True))
            try:
                raster = source.load()
            finally:
                source.close()
        if "band" in raster.dims:
            if raster.sizes["band"] != 1:
                raise RuntimeError(f"Earth Engine returned {raster.sizes['band']} bands; expected one")
            raster = raster.squeeze("band", drop=True)
        return raster.astype("float32")

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        raster = self._download_image(self._weekly_image(period_id), bbox)
        monday = np.datetime64(_week_monday(period_id).isoformat(), "D").astype("datetime64[ns]")
        result = raster.expand_dims(t=[monday]).to_dataset(name=self.output_variable)
        result.attrs.update(
            {
                "source_provider": "google_earth_engine",
                "source_dataset_id": _EARTH_ENGINE_ERA5_LAND_DAILY,
            }
        )
        return result


class ERA5LandTemperatureWeeklyPlugin(_DelegatingWeeklyPlugin):
    """ERA5-Land weekly mean 2-metre air temperature."""

    def __init__(self, **_: Any) -> None:
        """Build the weekly reducer around ERA5-Land daily mean temperature."""
        if _era5_daily_statistics_provider() == "earth_engine":
            delegate: BaseDatasetPlugin = _EarthEngineERA5LandWeeklyPlugin(
                source_band="temperature_2m",
                output_variable="t2m",
                reducer="mean",
                convert_kelvin=True,
            )
        else:
            source: BaseDatasetPlugin
            if _edh_credentials_available():
                source = _CachedERA5LandTempDaily(variable="t2m")
            else:
                source = _ERA5LandDailyStatisticPlugin("t2m", "mean")
            delegate = _WeeklyFromDailyPlugin(source, "t2m", "t2m", "mean")
        super().__init__(delegate)


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


class ERA5LandTemperatureMaxWeeklyPlugin(_DelegatingWeeklyPlugin):
    """ERA5-Land maximum 2-metre air temperature observed during a week."""

    def __init__(self, **_: Any) -> None:
        """Build the weekly reducer for maximum temperature."""
        if _era5_daily_statistics_provider() == "earth_engine":
            delegate: BaseDatasetPlugin = _EarthEngineERA5LandWeeklyPlugin(
                source_band="temperature_2m_max",
                output_variable="t2m_max",
                reducer="max",
                convert_kelvin=True,
            )
        else:
            source = _ERA5LandDailyStatisticPlugin("t2m", "max")
            delegate = _WeeklyFromDailyPlugin(source, "t2m", "t2m_max", "max")
        super().__init__(delegate)


class ERA5LandTemperatureMinWeeklyPlugin(_DelegatingWeeklyPlugin):
    """ERA5-Land minimum 2-metre air temperature observed during a week."""

    def __init__(self, **_: Any) -> None:
        """Build the weekly reducer for minimum temperature."""
        if _era5_daily_statistics_provider() == "earth_engine":
            delegate: BaseDatasetPlugin = _EarthEngineERA5LandWeeklyPlugin(
                source_band="temperature_2m_min",
                output_variable="t2m_min",
                reducer="min",
                convert_kelvin=True,
            )
        else:
            source = _ERA5LandDailyStatisticPlugin("t2m", "min")
            delegate = _WeeklyFromDailyPlugin(source, "t2m", "t2m_min", "min")
        super().__init__(delegate)


class ERA5LandDewpointTemperatureWeeklyPlugin(_DelegatingWeeklyPlugin):
    """ERA5-Land weekly mean 2-metre dew-point temperature."""

    def __init__(self, **_: Any) -> None:
        """Build the weekly reducer for mean dew-point temperature."""
        if _era5_daily_statistics_provider() == "earth_engine":
            delegate: BaseDatasetPlugin = _EarthEngineERA5LandWeeklyPlugin(
                source_band="dewpoint_temperature_2m",
                output_variable="d2m",
                reducer="mean",
                convert_kelvin=True,
            )
        else:
            source = _ERA5LandDailyStatisticPlugin("d2m", "mean")
            delegate = _WeeklyFromDailyPlugin(source, "d2m", "d2m", "mean")
        super().__init__(delegate)


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


class ERA5LandRelativeHumidityWeeklyPlugin(_DelegatingWeeklyPlugin):
    """ERA5-Land weekly mean relative humidity derived from temperature and dew point."""

    def __init__(self, **_: Any) -> None:
        """Build the weekly reducer for derived daily relative humidity."""
        if _era5_daily_statistics_provider() == "earth_engine":
            delegate: BaseDatasetPlugin = _EarthEngineERA5LandWeeklyPlugin(
                source_band=None,
                output_variable="hurs",
                reducer="mean",
                relative_humidity=True,
            )
        else:
            delegate = _WeeklyFromDailyPlugin(
                _ERA5LandDailyRelativeHumidityPlugin(),
                "hurs",
                "hurs",
                "mean",
            )
        super().__init__(delegate)


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
    url = os.getenv("ECMWF_DATASTORES_URL", "").strip() or os.getenv("CDSAPI_URL", "").strip() or _CDS_API_URL
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
                    datetime=(f"{search_start.isoformat()}T00:00:00Z/{effective_end.isoformat()}T23:59:59Z"),
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
        values = (raw * np.float32(0.0001)).where(((quality == 0) | (quality == 1)) & (raw >= -2000) & (raw <= 10000))

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
