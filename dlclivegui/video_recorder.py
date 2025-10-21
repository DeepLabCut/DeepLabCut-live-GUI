"""Video recording support using the vidgear library."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    from vidgear.gears import WriteGear
except ImportError:  # pragma: no cover - handled at runtime
    WriteGear = None  # type: ignore[assignment]


class VideoRecorder:
    """Thin wrapper around :class:`vidgear.gears.WriteGear`."""

    def __init__(self, output: Path | str, options: Optional[Dict[str, Any]] = None):
        self._output = Path(output)
        self._options = options or {}
        self._writer: Optional[WriteGear] = None

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
        self._output.parent.mkdir(parents=True, exist_ok=True)
        self._writer = WriteGear(output_filename=str(self._output), logging=False, **self._options)

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            return
        self._writer.write(frame)

    def stop(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
