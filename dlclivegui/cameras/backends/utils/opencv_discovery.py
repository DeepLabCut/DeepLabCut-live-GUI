#  dlclivegui/cameras/backends/utils/opencv_discovery.py
from __future__ import annotations

import logging
import platform
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2

logger = logging.getLogger(__name__)


def _aspect(w: int, h: int) -> float:
    return (float(w) / float(h)) if (w > 0 and h > 0) else 0.0


def _aspect_close(a: float, b: float, tol: float) -> bool:
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) / b <= tol


@dataclass(frozen=True)
class ModeRequest:
    width: int
    height: int
    fps: float = 0.0
    enforce_aspect: str = "strict"  # strict|prefer|ignore
    aspect_tol: float = 0.01  # 1% relative aspect tolerance
    area_tol: float = 0.05  # accept 5% area mismatch as “close enough”


@dataclass(frozen=True)
class ModeResult:
    width: int
    height: int
    fps: float
    accepted: bool
    notes: str = ""


@dataclass(frozen=True)
class CameraCandidate:
    """
    Normalized camera record used by our backend.

    - index/backend: OpenCV-friendly identifiers (backend-aware index is best).
    - name/path/vid/pid: metadata useful for stable selection and diagnostics.
    """

    index: int
    backend: int
    name: str = ""
    path: str = ""
    vid: int | None = None
    pid: int | None = None

    @property
    def stable_id(self) -> str:
        """
        Best-effort stable ID for caching / selection.
        Path/uniqueID is generally most stable; fall back to VID:PID + name.
        """
        if self.path:
            return f"path:{self.path}"
        if self.vid is not None and self.pid is not None:
            return f"usb:{self.vid:04x}:{self.pid:04x}:{self.name}"
        return f"name:{self.name}:idx:{self.index}:b:{self.backend}"


def _try_import_enumerator():
    """
    Optional dependency: cv2-enumerate-cameras.
    If unavailable, return None (caller can fallback).
    """
    try:
        from cv2_enumerate_cameras import enumerate_cameras  # type: ignore

        return enumerate_cameras
    except Exception:
        return None


def list_cameras(
    api_preference: int | None = None,
    enumerator: Callable[..., Sequence[Any]] | None = None,
) -> list[CameraCandidate]:
    """
    Enumerate cameras using cv2-enumerate-cameras if installed.
    Returns a list of CameraCandidate (possibly empty).
    """
    enum_fn = enumerator or _try_import_enumerator()
    if enum_fn is None:
        logger.debug("cv2-enumerate-cameras not installed; cannot enumerate cameras.")
        return []

    if api_preference is None:
        api_preference = cv2.CAP_ANY

    cams = []
    try:
        for info in enum_fn(api_preference):
            # cv2-enumerate-cameras CameraInfo typically has: index, backend, name, path, vid, pid
            cams.append(
                CameraCandidate(
                    index=int(info.index),
                    backend=int(info.backend),
                    name=str(getattr(info, "name", "") or ""),
                    path=str(getattr(info, "path", "") or ""),
                    vid=getattr(info, "vid", None),
                    pid=getattr(info, "pid", None),
                )
            )
    except Exception as exc:
        logger.debug("Camera enumeration failed: %s", exc)
        return []

    return cams


def select_camera(
    cameras: Sequence[CameraCandidate],
    *,
    prefer_stable_id: str | None = None,
    prefer_name_substr: str | None = None,
    prefer_vid_pid: tuple[int, int] | None = None,
    fallback_index: int | None = None,
) -> CameraCandidate | None:
    """
    Choose a camera deterministically from the list.

    Selection order:
    1) stable_id exact match (best for caching)
    2) VID/PID match
    3) name contains substring (case-insensitive)
    4) fallback_index (by backend-aware index)
    5) first camera

    This is intentionally simple and testable.
    """
    if not cameras:
        return None

    if prefer_stable_id:
        for c in cameras:
            if c.stable_id == prefer_stable_id:
                return c

    if prefer_vid_pid:
        v, p = prefer_vid_pid
        for c in cameras:
            if c.vid == v and c.pid == p:
                return c

    if prefer_name_substr:
        needle = prefer_name_substr.lower()
        for c in cameras:
            if needle in (c.name or "").lower():
                return c

    if fallback_index is not None:
        for c in cameras:
            if c.index == fallback_index:
                return c

    return cameras[0]


@dataclass(frozen=True)
class OpenSpec:
    """What we need to open the camera reliably with OpenCV."""

    index: int
    backend: int  # cv2.CAP_*
    used_fallback: bool = False


def preferred_backend_for_platform() -> int:
    sys = platform.system()
    if sys == "Windows":
        return getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
    if sys == "Darwin":
        return getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)


def try_open(index: int, backend: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        return cap
    try:
        cap.release()
    except Exception:
        pass
    return None


def open_with_fallbacks(index: int, backend: int) -> tuple[cv2.VideoCapture | None, OpenSpec]:
    """
    Try (index, backend) first, then platform-specific fallbacks.
    This is isolated from your backend class so it’s easy to test and evolve.
    """
    cap = try_open(index, backend)
    if cap:
        return cap, OpenSpec(index=index, backend=backend, used_fallback=False)

    sys = platform.system()
    # Windows: DSHOW -> MSMF -> ANY
    if sys == "Windows":
        msmf = getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)
        if backend != msmf:
            cap = try_open(index, msmf)
            if cap:
                return cap, OpenSpec(index=index, backend=msmf, used_fallback=True)

    # Generic fallback
    cap = try_open(index, cv2.CAP_ANY)
    if cap:
        return cap, OpenSpec(index=index, backend=cv2.CAP_ANY, used_fallback=True)

    return None, OpenSpec(index=index, backend=backend, used_fallback=True)


def generate_candidates(req_w: int, req_h: int, enforce_aspect: str) -> list[tuple[int, int]]:
    """
    Generate a bounded set of candidate resolutions near the requested size.
    No device assumptions: just proximity + common aspect-preserving steps.
    """
    req_aspect = _aspect(req_w, req_h)

    # Near-scale factors: try exact, then slightly down/up
    scales = [1.0, 0.9, 0.8, 0.75, 0.67, 1.1, 1.25]
    candidates: list[tuple[int, int]] = []

    def snap(x: float) -> int:
        # Snap to multiples of 8 (common constraint); keep >= 2
        v = int(round(x))
        v = max(2, v - (v % 8))
        return v

    for s in scales:
        w = snap(req_w * s)
        h = snap(req_h * s)
        if w > 0 and h > 0:
            candidates.append((w, h))

    # Add a few common standards *matching the requested aspect family*
    # (This is not hardware-specific; it’s format-common.)
    if enforce_aspect in ("strict", "prefer"):
        if _aspect_close(req_aspect, 4 / 3, 0.02):
            candidates += [(640, 480), (800, 600), (1024, 768), (1280, 960)]
        elif _aspect_close(req_aspect, 16 / 9, 0.02):
            candidates += [(640, 360), (960, 540), (1280, 720), (1920, 1080)]

    # Deduplicate preserving order
    seen = set()
    out = []
    for w, h in candidates:
        if (w, h) not in seen:
            out.append((w, h))
            seen.add((w, h))
    return out


def apply_mode_with_verification(
    cap: cv2.VideoCapture,
    request: ModeRequest,
    *,
    candidates: Iterable[tuple[int, int]] | None = None,
    warmup_grabs: int = 3,
) -> ModeResult:
    """
    Attempt to configure the camera as close as possible to request.

    Returns ModeResult(accepted=True) if we achieved a “close enough” match based on policy.
    """
    req_w, req_h = int(request.width), int(request.height)
    req_fps = float(request.fps or 0.0)
    req_aspect = _aspect(req_w, req_h)

    cand_list = (
        list(candidates) if candidates is not None else generate_candidates(req_w, req_h, request.enforce_aspect)
    )

    best: ModeResult | None = None
    best_score = float("inf")

    for w, h in cand_list:
        # If strict aspect: skip obviously wrong aspect early
        if request.enforce_aspect == "strict":
            if not _aspect_close(_aspect(w, h), req_aspect, request.aspect_tol):
                continue

        # Set W/H
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))

        # Set FPS if requested
        if req_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, float(req_fps))

        # Warm up (some backends only apply after a few grabs)
        for _ in range(max(0, warmup_grabs)):
            cap.grab()

        # Read back
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        afps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        # Compute closeness
        a_aspect = _aspect(aw, ah)
        aspect_err = abs(a_aspect - req_aspect)
        area_err = abs((aw * ah) - (req_w * req_h)) / max(1.0, float(req_w * req_h))
        fps_err = 0.0 if req_fps <= 0 or afps <= 0 else abs(afps - req_fps) / max(1.0, req_fps)

        # Heavy weight on aspect unless ignore
        aspect_weight = 10.0 if request.enforce_aspect != "ignore" else 0.5
        score = aspect_weight * aspect_err + 3.0 * area_err + 1.0 * fps_err

        accepted = True
        notes = []

        if request.enforce_aspect == "strict":
            if not _aspect_close(a_aspect, req_aspect, request.aspect_tol):
                accepted = False
                notes.append("aspect_mismatch")

        if area_err > request.area_tol:
            # area mismatch alone isn't fatal if aspect is fine; mark note
            notes.append(f"area_err={area_err:.3f}")

        if req_fps > 0 and afps > 0 and abs(afps - req_fps) > max(1.0, 0.05 * req_fps):
            notes.append(f"fps_err={fps_err:.3f}")

        result = ModeResult(width=aw, height=ah, fps=afps, accepted=accepted, notes=",".join(notes))

        # Track best (even if not accepted) to provide a useful fallback
        if score < best_score:
            best_score = score
            best = result

        if accepted:
            return result

    return best or ModeResult(width=0, height=0, fps=0.0, accepted=False, notes="no_candidates")
