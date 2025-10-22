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
        frame_size: Optional[Tuple[int, int]] = None,
        frame_rate: Optional[float] = None,
        codec: str = "libx264",
        crf: int = 23,
    ):
        self._output = Path(output)
        self._writer: Optional[WriteGear] = None
        self._frame_size = frame_size
        self._frame_rate = frame_rate
        self._codec = codec
        self._crf = int(crf)

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
        fps_value = float(self._frame_rate) if self._frame_rate else 30.0

        writer_kwargs: Dict[str, Any] = {
            "compression_mode": True,
            "logging": True,
            "-input_framerate": fps_value,
            "-vcodec": (self._codec or "libx264").strip() or "libx264",
            "-crf": int(self._crf),
        }
        # TODO deal with pixel format

        self._output.parent.mkdir(parents=True, exist_ok=True)
        self._writer = WriteGear(output=str(self._output), **writer_kwargs)

    def configure_stream(
        self, frame_size: Tuple[int, int], frame_rate: Optional[float]
    ) -> None:
        self._frame_size = frame_size
        self._frame_rate = frame_rate

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            return
        if frame.dtype != np.uint8:
            frame_float = frame.astype(np.float32, copy=False)
            max_val = float(frame_float.max()) if frame_float.size else 0.0
            scale = 1.0
            if max_val > 0:
                scale = 255.0 / max_val if max_val > 255.0 else (255.0 if max_val <= 1.0 else 1.0)
            frame = np.clip(frame_float * scale, 0.0, 255.0).astype(np.uint8)
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        frame = np.ascontiguousarray(frame)
        try:
            self._writer.write(frame)
        except OSError as exc:
            writer = self._writer
            self._writer = None
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            raise RuntimeError(f"Video encoding failed: {exc}") from exc

    def stop(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
