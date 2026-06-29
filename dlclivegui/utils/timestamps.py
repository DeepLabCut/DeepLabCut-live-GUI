# dlclivegui/utils/timestamps.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrameTimestampMetadata:
    """Optional backend-provided timestamp metadata for a captured frame.

    This supplements, but does not replace, the software timestamp.

    Notes:
        - `seconds` is in the backend/hardware timebase.
        - `wall_clock_time` should only be set if the backend can confidently
          provide Unix/wall-clock seconds.
        - `raw_value` preserves the original device-specific timestamp.
    """

    source: str
    backend: str

    # Which value should downstream consumers use by default, if any.
    # Expected values: "seconds", "wall_clock_time", or "raw_value".
    default_reported: str | None = None

    # Device/hardware timebase value, if convertible to seconds
    seconds: float | None = None

    # True Unix/wall-clock timestamp, if available
    wall_clock_time: float | None = None

    # Raw backend value, e.g. device clock ticks
    raw_value: int | float | str | None = None
    raw_unit: str | None = None

    # Conversion metadata.
    tick_frequency_hz: float | None = None
    timebase: str | None = None

    # e.g. "camera_clock", "ptp_camera_clock", "hardware_wall_clock",
    # "frame_counter", "unknown"
    kind: str = "unknown"

    # Backend-specific extras.
    extra: dict[str, Any] | None = None

    def to_source_dict(self) -> dict[str, Any]:
        """Return metadata that should be written once per recording stream."""
        return {
            "source": self.source,
            "backend": self.backend,
            "default_reported": self.default_reported,
            "raw_unit": self.raw_unit,
            "tick_frequency_hz": self.tick_frequency_hz,
            "timebase": self.timebase,
            "kind": self.kind,
            "extra": self.extra or {},
        }

    def to_frame_dict(self) -> dict[str, Any]:
        """Return defined per-frame timestamp values only."""
        ts = {}
        for k in ["seconds", "wall_clock_time", "raw_value"]:
            v = getattr(self, k)
            if v is not None:
                ts[k] = v
        return ts

    def to_dict(self) -> dict[str, Any]:
        """Return full representation, useful for logging/debugging."""
        return {
            **self.to_source_dict(),
            **self.to_frame_dict(),
        }

    def get_default_reported(self) -> int | float | str | None:
        """Return the value selected by `default_reported`, if configured."""
        if not self.default_reported:
            return None
        return self.to_frame_dict().get(self.default_reported)
