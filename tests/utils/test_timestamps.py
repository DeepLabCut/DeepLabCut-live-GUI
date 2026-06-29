import pytest

from dlclivegui.utils.timestamps import FrameTimestampMetadata


class TestFrameTimestampMetadata:
    def test_splits_source_and_frame_values(self):
        meta = FrameTimestampMetadata(
            source="grab_result.GetTimeStamp",
            backend="basler",
            default_reported="seconds",
            seconds=0.123456789,
            wall_clock_time=None,
            raw_value=123456789,
            raw_unit="ticks",
            tick_frequency_hz=1_000_000_000.0,
            timebase="Basler camera timestamp counter",
            kind="camera_clock",
        )

        assert meta.to_source_dict() == {
            "source": "grab_result.GetTimeStamp",
            "backend": "basler",
            "default_reported": "seconds",
            "raw_unit": "ticks",
            "tick_frequency_hz": 1_000_000_000.0,
            "timebase": "Basler camera timestamp counter",
            "kind": "camera_clock",
            "extra": {},
        }

        frame_dict = meta.to_frame_dict()
        assert frame_dict["seconds"] == pytest.approx(0.123456789)
        assert frame_dict["raw_value"] == 123456789
        assert "wall_clock_time" not in frame_dict

        assert meta.get_default_reported() == pytest.approx(0.123456789)

    def test_default_reported_raw_value(self):
        meta = FrameTimestampMetadata(
            source="device_counter",
            backend="some_backend",
            default_reported="raw_value",
            raw_value=42,
            raw_unit="frames",
            kind="frame_counter",
        )

        assert meta.to_frame_dict() == {"raw_value": 42}
        assert meta.get_default_reported() == 42

    def test_unknown_default_field_returns_none(self):
        meta = FrameTimestampMetadata(
            source="device_counter",
            backend="some_backend",
            default_reported="seconds",
            raw_value=42,
            raw_unit="frames",
            kind="frame_counter",
        )

        assert meta.to_frame_dict() == {"raw_value": 42}
        assert meta.get_default_reported() is None
