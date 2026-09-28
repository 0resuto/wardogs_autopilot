"""Unit tests for Pydantic configuration schema and logging setup."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from autopilot.common.config import (
    AppConfig,
    CaptureConfig,
    LocatorConfig,
    resolve_config_path,
)
from autopilot.common.log import get_logger, setup_logging

ROOT = Path(__file__).resolve().parents[1]


def test_relative_config_path_is_rooted_at_project_root():
    """A relative --config path always points into the project root."""
    assert resolve_config_path("config.json") == ROOT / "config.json"

    absolute = ROOT / "output" / "other.json"
    assert resolve_config_path(absolute) == absolute


def test_config_roundtrip_to_custom_path(tmp_path):
    """load/save keep using the same (custom) file instead of config.json."""
    target = tmp_path / "custom.json"
    AppConfig().save(target)

    cfg = AppConfig.load(target)
    assert Path(cfg.cfg_path) == target

    cfg.capture.fps = 33
    cfg.save()
    assert AppConfig.load(target).capture.fps == 33


def test_app_config_load():
    """Verify loading and validating the root config.json."""
    cfg = AppConfig.load("config.json")
    assert cfg.capture.fps > 0
    assert len(cfg.capture.mmap_roi) == 4
    assert cfg.map.name != ""
    assert cfg.navigator.port.startswith("COM") or cfg.navigator.key_source != "arduino"

    data = cfg.to_dict()
    assert isinstance(data, dict)
    assert "capture" in data
    assert "locator" in data


def test_invalid_roi_validation():
    """Verify mmap_roi validation bounds."""
    with pytest.raises(ValidationError):
        CaptureConfig(mmap_roi=[0, 0, 5, 5])  # Too small (<10x10)

    with pytest.raises(ValidationError):
        CaptureConfig(mmap_roi=[0, 0, 100])  # Length != 4


def test_speed_roi_validation():
    """Verify the optional speedometer ROI accepts None and validated boxes."""
    assert CaptureConfig(speed_roi=None).speed_roi is None
    assert CaptureConfig(speed_roi=[1, 2, 40, 30]).speed_roi == [1, 2, 40, 30]

    with pytest.raises(ValidationError):
        CaptureConfig(speed_roi=[0, 0, 5, 5])  # Too small (<8x8)

    with pytest.raises(ValidationError):
        CaptureConfig(speed_roi=[0, 0, 100])  # Length != 4


def test_invalid_locator_ratio():
    """Verify ratio bounds on LocatorConfig."""
    with pytest.raises(ValidationError):
        LocatorConfig(ratio=1.5)  # Ratio > 1.0


def test_logging_setup(tmp_path):
    """Verify logger writes messages to file."""
    setup_logging(log_dir=str(tmp_path), log_filename="test_run.log")
    logger = get_logger("test_module")
    logger.info("Test log message")

    log_file = tmp_path / "test_run.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test log message" in content
    assert "[INFO]" in content
    assert "[test_module]" in content


def test_follow_driver_with_typed_config():
    """Verify FollowDriver initializes cleanly with typed NavigatorConfig."""
    from autopilot.common.config import NavigatorConfig
    from autopilot.navigation.follow import FollowDriver

    nav_cfg = NavigatorConfig(arrive_r=42.0, slow_r=250.0, speed_cap_kmh=65.0)
    driver = FollowDriver(loc=None, pts=[(0, 0), (100, 100)], nav_cfg=nav_cfg, kb=None)
    assert driver.arrive_r == 42.0
    assert driver.slow_r == 250.0
    assert driver.speed_cap_kmh == 65.0
    assert isinstance(driver.nav_cfg, NavigatorConfig)


def test_live_locator_with_typed_config():
    """Verify LiveLocator handles AppConfig directly."""
    from autopilot.common.config import AppConfig, CaptureConfig, LocatorConfig
    from autopilot.vision.tracker import LiveLocator

    cfg = AppConfig()
    locator = LiveLocator(cfg=cfg, mask=None)
    assert isinstance(locator.app_cfg, AppConfig)
    assert isinstance(locator.cap_cfg, CaptureConfig)
    assert isinstance(locator.loc_cfg, LocatorConfig)


def test_map_store_typed_cfg_bridge():
    """Verify MapStore provides validated config dicts from AppConfig."""
    from autopilot.vision.map_store import MapStore

    store = MapStore()
    loc_cfg = store.loc_cfg()
    map_cfg = store.map_cfg()
    assert isinstance(loc_cfg, dict)
    assert "local_radius" in loc_cfg
    assert isinstance(map_cfg, dict)
    assert "gray_conv" in map_cfg
