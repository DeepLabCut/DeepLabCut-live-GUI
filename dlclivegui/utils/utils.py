from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SUPPORTED_MODELS = [".pt", ".pth", ".pb"]
_INVALID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def is_model_file(file_path: Path | str) -> bool:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if not file_path.is_file():
        return False
    return file_path.suffix.lower() in SUPPORTED_MODELS


def sanitize_name(name: str, *, fallback: str = "session") -> str:
    """Make a user-provided string safe for filesystem paths."""
    name = (name or "").strip()
    if not name:
        return fallback
    name = _INVALID_CHARS.sub("_", name)
    name = name.strip("._- ")
    return name or fallback


def timestamp_string(*, with_ms: bool = True) -> str:
    """Timestamp suitable for filenames/folders."""
    now = datetime.now()
    if with_ms:
        # YYYYMMDD_HHMMSS_mmm
        return now.strftime("%Y%m%d_%H%M%S_%f")[:19]
    return now.strftime("%Y%m%d_%H%M%S")


def split_stem_ext(base_filename: str, container: str) -> tuple[str, str]:
    """
    Decide final stem/ext using user input + container dropdown.
    If user typed an extension, keep it. Else use container.
    """
    base = (base_filename or "").strip()
    container = (container or "mp4").strip().lstrip(".") or "mp4"

    if not base:
        base = "recording"

    p = Path(base)
    if p.suffix:
        return p.stem, p.suffix.lstrip(".")
    return base, container


def next_run_index(session_dir: Path, *, prefix: str = "run_") -> int:
    """Find next available run index (run_0001, run_0002, ...)."""
    existing = []
    for child in session_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                existing.append(int(suffix))
    return (max(existing) + 1) if existing else 1


def build_run_dir(session_dir: Path, *, use_timestamp: bool) -> Path:
    """
    Build output path for session as: session_dir/<run-folder>/...
    run-folder is always unique:
      - timestamp-based if use_timestamp
      - incrementing run_0001 if not
    """
    session_dir.mkdir(parents=True, exist_ok=True)

    if use_timestamp:
        run_name = f"run_{timestamp_string(with_ms=True)}"
        run_dir = session_dir / run_name
        # Unlikely collision, but guard anyway:
        if run_dir.exists():
            run_dir = session_dir / f"{run_name}_{timestamp_string(with_ms=True)}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    idx = next_run_index(session_dir, prefix="run_")
    run_dir = session_dir / f"run_{idx:04d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


@dataclass(frozen=True)
class RecordingPlan:
    session_dir: Path
    run_dir: Path
    files_by_camera_id: dict[str, Path]


def build_recording_plan(
    *,
    output_dir: str | Path,
    session_name: str,
    base_filename: str,
    container: str,
    camera_ids: Iterable[str],
    use_timestamp: bool,
) -> RecordingPlan:
    """
    Construct recording plan with stable filenames per camera.
    Directory structure:
      output_dir/session_name/run_xxxx/  (or run_timestamp/)
      files: {stem}_{cam}.ext  (stable filenames, no timestamp needed)
    """
    output_dir = Path(output_dir).expanduser().resolve()
    session = sanitize_name(session_name, fallback="session")
    session_dir = output_dir / session
    run_dir = build_run_dir(session_dir, use_timestamp=use_timestamp)

    stem, ext = split_stem_ext(base_filename, container)
    stem = sanitize_name(stem, fallback="recording")

    files: dict[str, Path] = {}
    for cam_id in camera_ids:
        safe_cam = sanitize_name(cam_id.replace(":", "_"), fallback="cam")
        path = run_dir / f"{stem}_{safe_cam}.{ext}"
        files[cam_id] = path

    return RecordingPlan(session_dir=session_dir, run_dir=run_dir, files_by_camera_id=files)


class FPSTracker:
    """Track per-camera FPS within a sliding time window."""

    def __init__(self, window_seconds: float = 5.0, maxlen: int = 240):
        self.window_seconds = window_seconds
        self._times: dict[str, deque[float]] = {}
        self._maxlen = maxlen

    def clear(self) -> None:
        self._times.clear()

    def note_frame(self, camera_id: str) -> None:
        now = time.perf_counter()
        dq = self._times.get(camera_id)
        if dq is None:
            dq = deque(maxlen=self._maxlen)
            self._times[camera_id] = dq
        dq.append(now)
        while dq and (now - dq[0]) > self.window_seconds:
            dq.popleft()

    def fps(self, camera_id: str) -> float:
        dq = self._times.get(camera_id)
        if not dq or len(dq) < 2:
            return 0.0
        duration = dq[-1] - dq[0]
        if duration <= 0:
            return 0.0
        return (len(dq) - 1) / duration
