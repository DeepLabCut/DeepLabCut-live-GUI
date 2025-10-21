"""Video recording support using the vidgear library."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from vidgear.gears import WriteGear
except ImportError:  # pragma: no cover - handled at runtime
    WriteGear = None  # type: ignore[assignment]


class VideoRecorder:
    """Thin wrapper around :class:`vidgear.gears.WriteGear`."""

    def __init__(
        self,
        output: Path | str,
        options: Optional[Dict[str, Any]] = None,
        frame_size: Optional[Tuple[int, int]] = None,
        frame_rate: Optional[float] = None,
    ):
        self._output = Path(output)
        self._options = options or {}
        self._writer: Optional[WriteGear] = None
        self._frame_size = frame_size
        self._frame_rate = frame_rate

    @property
    def is_running(self) -> bool:
        return self._writer is not None

    def start(self) -> None:
        if WriteGear is None:
            raise RuntimeError(
                "vidgear is required for video recording. Install it with 'pip install vidgear'."
            )
        if self._writer is not None:
            return
        options = dict(self._options)
        if self._frame_size and "resolution" not in options:
            options["resolution"] = tuple(int(x) for x in self._frame_size)
        if self._frame_rate and "frame_rate" not in options:
            options["frame_rate"] = float(self._frame_rate)
        self._output.parent.mkdir(parents=True, exist_ok=True)
        self._writer = WriteGear(output=str(self._output), logging=False, **options)

    def configure_stream(
        self, frame_size: Tuple[int, int], frame_rate: Optional[float]
    ) -> None:
        self._frame_size = frame_size
        self._frame_rate = frame_rate

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            return
        self._writer.write(frame)

    def stop(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
