from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "plugins"))

from datasets.drc_weekly import (  # noqa: E402
    ERA5HeatWeeklyPlugin,
    MODISVegetationWeeklyPlugin,
    _complete_week_ids,
    _configure_ecmwf_datastores,
    _ERA5LandDailyRelativeHumidityPlugin,
    _overlap_days,
    _WeeklyFromDailyPlugin,
)

from open_climate_service.streaming import BaseDatasetPlugin  # noqa: E402


class _DailySource(BaseDatasetPlugin):
    def __init__(self, variable: str, value: float | None = None) -> None:
        self.variable = variable
        self.value = value

    async def periods(self, start: str, end: str) -> list[str]:
        current = date.fromisoformat(start)
        limit = date.fromisoformat(end)
        result = []
        while current <= limit:
            result.append(current.isoformat())
            current += timedelta(days=1)
        return result

    def fetch_period(self, period_id: str, bbox: list[float], **params: object) -> xr.Dataset:
        value = self.value if self.value is not None else float(date.fromisoformat(period_id).day)
        return xr.Dataset(
            {self.variable: (("t", "y", "x"), np.array([[[value]]], dtype=np.float32))},
            coords={"t": [np.datetime64(period_id)], "y": [0.0], "x": [0.0]},
        )


def test_complete_week_ids_excludes_partial_week() -> None:
    assert _complete_week_ids("2026-W01", "2026-W03", date(2026, 1, 17)) == ["2026-W01", "2026-W02"]


def test_weekly_daily_reducer_requires_all_seven_days() -> None:
    plugin = _WeeklyFromDailyPlugin(_DailySource("value"), "value", "weekly", "sum")

    assert asyncio.run(plugin.periods("2026-W01", "2026-W01")) == ["2026-W01"]
    result = plugin.fetch_period("2026-W01", [0, 0, 1, 1])

    assert result.weekly.item() == pytest.approx(100.0)
    assert str(result.t.values[0])[:10] == "2025-12-29"


def test_relative_humidity_is_derived_and_bounded() -> None:
    plugin = _ERA5LandDailyRelativeHumidityPlugin()
    plugin.temperature = _DailySource("t2m", 30.0)  # type: ignore[assignment]
    plugin.dewpoint = _DailySource("d2m", 20.0)  # type: ignore[assignment]

    result = plugin.fetch_period("2026-01-01", [0, 0, 1, 1])

    assert result.hurs.item() == pytest.approx(55.1, abs=0.2)
    assert 0.0 <= result.hurs.item() <= 100.0


def test_era5_heat_reduces_hourly_kelvin_to_weekly_celsius() -> None:
    times = np.arange(np.datetime64("2026-01-05T00"), np.datetime64("2026-01-12T00"), np.timedelta64(1, "h"))
    source = xr.Dataset(
        {
            "utci": (
                ("time", "latitude", "longitude"),
                np.full((len(times), 2, 2), 303.15, dtype=np.float32),
                {"units": "K"},
            )
        },
        coords={"time": times, "latitude": [5.0, -5.0], "longitude": [15.0, 25.0]},
    )
    plugin = ERA5HeatWeeklyPlugin(statistic="mean")
    plugin._source = source

    result = plugin.fetch_period("2026-W02", [14.0, -6.0, 26.0, 6.0])

    assert result.utci.mean().item() == pytest.approx(30.0, abs=1e-3)
    assert result.utci.dims == ("t", "y", "x")


def test_modis_overlap_counts_inclusive_days() -> None:
    assert _overlap_days(date(2026, 1, 1), date(2026, 1, 16), date(2026, 1, 5), date(2026, 1, 11)) == 7
    assert _overlap_days(date(2026, 1, 1), date(2026, 1, 4), date(2026, 1, 5), date(2026, 1, 11)) == 0


def test_modis_loads_quality_band_as_int16(monkeypatch: pytest.MonkeyPatch) -> None:
    item = SimpleNamespace(
        id="MOD13A1.A2026273.h20v09.061.test",
        properties={"platform": "terra"},
        datetime=None,
        common_metadata=SimpleNamespace(
            start_datetime=datetime(2026, 9, 30, tzinfo=UTC),
            end_datetime=datetime(2026, 10, 15, tzinfo=UTC),
        ),
    )

    class _Search:
        def items(self) -> list[object]:
            return [item]

    class _Catalog:
        def search(self, **_: object) -> _Search:
            return _Search()

    captured: dict[str, object] = {}

    def fake_load(_: object, **kwargs: object) -> xr.Dataset:
        captured.update(kwargs)
        return xr.Dataset(
            {
                "500m_16_days_NDVI": (("time", "y", "x"), np.array([[[5000]]], dtype=np.int16)),
                "500m_16_days_pixel_reliability": (
                    ("time", "y", "x"),
                    np.array([[[0]]], dtype=np.int16),
                ),
            },
            coords={"time": [np.datetime64("2026-09-30")], "y": [0.0], "x": [0.0]},
        )

    monkeypatch.setattr("odc.stac.load", fake_load)
    plugin = MODISVegetationWeeklyPlugin(index="ndvi")
    plugin._client = _Catalog()

    result = plugin.fetch_period("2026-W40", [20.0, -5.0, 20.1, -4.9])

    assert captured["dtype"] == {
        "500m_16_days_NDVI": "int16",
        "500m_16_days_pixel_reliability": "int16",
    }
    assert result.ndvi.item() == pytest.approx(0.5)


def test_cdsapi_key_is_bridged_to_ecmwf_datastores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CDSAPI_KEY", "secret-test-token")
    monkeypatch.delenv("ECMWF_DATASTORES_URL", raising=False)
    monkeypatch.delenv("ECMWF_DATASTORES_KEY", raising=False)

    url, token = _configure_ecmwf_datastores()

    assert url == "https://cds.climate.copernicus.eu/api"
    assert token == "secret-test-token"
    assert os.environ["ECMWF_DATASTORES_KEY"] == "secret-test-token"


def test_all_import_pipeline_dataset_ids_have_templates() -> None:
    repository = Path(__file__).parents[1]
    templates = yaml.safe_load((repository / "plugins/datasets/drc_weekly.yaml").read_text(encoding="utf-8"))
    template_ids = {item["id"] for item in templates}
    expected = {
        "chirps3_precipitation_weekly",
        "era5land_precipitation_weekly",
        "era5land_temperature_weekly",
        "era5land_temperature_max_weekly",
        "era5land_temperature_min_weekly",
        "era5land_dewpoint_temperature_weekly",
        "era5land_relative_humidity_weekly",
        "era5heat_utci_weekly",
        "era5heat_utci_max_weekly",
        "era5heat_utci_min_weekly",
        "modis_ndvi_weekly",
        "modis_evi_weekly",
    }

    assert template_ids == expected
