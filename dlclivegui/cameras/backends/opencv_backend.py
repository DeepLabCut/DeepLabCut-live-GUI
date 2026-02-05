"""OpenCV-based camera backend (platform-optimized, fast startup, robust read)."""

# dlclivegui/cameras/backends/opencv_backend.py
from __future__ import annotations

import logging
import os
import platform
import time
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator

from ..base import CameraBackend, register_backend
from .utils.opencv_discovery import (
    ModeRequest,
    apply_mode_with_verification,
    list_cameras,
    open_with_fallbacks,
    select_camera,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # FIXME @C-Achard remove before release

if TYPE_CHECKING:
    from dlclivegui.config import CameraSettings


AspectPolicy = Literal["strict", "prefer", "ignore"]
FourCC = Literal["MJPG", "YUY2", "NV12", "H264", "XRGB", "BGR3"]  # expand as needed


class OpenCVOptions(BaseModel):
    # --- device selection ---
    device_id: str | None = None  # stable_id from cv2-enumerate-cameras
    device_name: str | None = None  # substring match
    device_vid: int | None = None
    device_pid: int | None = None

    # --- backend/open behavior ---
    api: str | None = None  # "DSHOW", "MSMF", "V4L2", "AVFOUNDATION", "ANY"
    fast_start: bool = False
    alt_index_probe: bool = False

    # --- format negotiation policy ---
    enforce_aspect: AspectPolicy = "strict"
    aspect_tol: float = Field(default=0.01, ge=0.0, le=0.2)  # 1% default
    area_tol: float = Field(default=0.05, ge=0.0, le=1.0)  # 5% default

    # --- codec policy ---
    prefer_mjpg: bool = False  # opt-in MJPG attempt on Windows
    fourcc: FourCC | None = None  # explicit request overrides prefer_mjpg

    @model_validator(mode="after")
    def _codec_consistency(self):
        # If user explicitly sets fourcc, we don't need prefer_mjpg
        if self.fourcc is not None and self.prefer_mjpg:
            self.prefer_mjpg = False
        return self


@register_backend("opencv")
class OpenCVCameraBackend(CameraBackend):
    """
    Platform-aware OpenCV backend:

    - Windows: prefer DSHOW, fall back to MSMF/ANY.
      Order: FOURCC -> resolution -> FPS. Try standard UVC modes if request fails.
      Optional alt-index probe (index+1) for Logitech-like endpoints: properties["alt_index_probe"]=True
      Optional fast-start: properties["fast_start"]=True

    - macOS: prefer AVFOUNDATION, fall back to ANY.

    - Linux: prefer V4L2, fall back to GStreamer (if explicitly requested) or ANY.
      Discovery can use /dev/video* to avoid blind opens (via quick_ping()).

    Robust read(): returns (None, ts) on transient failures (never raises).
    """

    OPTIONS_KEY = "opencv"
    SAFE_PROP_IDS = {
        int(getattr(cv2, "CAP_PROP_EXPOSURE", 15)),
        int(getattr(cv2, "CAP_PROP_AUTO_EXPOSURE", 21)),
        int(getattr(cv2, "CAP_PROP_GAIN", 14)),
        int(getattr(cv2, "CAP_PROP_FPS", 5)),
        int(getattr(cv2, "CAP_PROP_BRIGHTNESS", 10)),
        int(getattr(cv2, "CAP_PROP_CONTRAST", 11)),
        int(getattr(cv2, "CAP_PROP_SATURATION", 12)),
        int(getattr(cv2, "CAP_PROP_HUE", 13)),
        int(getattr(cv2, "CAP_PROP_CONVERT_RGB", 17)),
    }

    def __init__(self, settings):
        super().__init__(settings)
        self._capture: cv2.VideoCapture | None = None
        self._resolution: tuple[int, int] = self._parse_resolution(settings.properties.get("resolution"))
        opt = self.parse_options(settings)
        self._fast_start: bool = opt.fast_start
        self._alt_index_probe: bool = opt.alt_index_probe
        self._actual_width: int | None = None
        self._actual_height: int | None = None
        self._actual_fps: float | None = None
        self._codec_str: str = ""
        self._mjpg_attempted: bool = False

    @classmethod
    def parse_options(cls, settings: CameraSettings) -> OpenCVOptions:
        raw = (settings.properties or {}).get(cls.OPTIONS_KEY, {})
        # no flat keys supported — clean ground
        return OpenCVOptions.model_validate(raw)

    @classmethod
    def options_schema(cls) -> dict:
        return OpenCVOptions.model_json_schema()

    # ----------------------------
    # Public API
    # ----------------------------
    def open(self) -> None:
        opt = self.parse_options(self.settings)  # typed + validated
        backend_flag = self._preferred_backend_flag(opt.api)
        index = int(self.settings.index)

        cams = list_cameras(backend_flag)
        chosen = select_camera(
            cams,
            prefer_stable_id=opt.device_id,
            prefer_name_substr=opt.device_name,
            prefer_vid_pid=(opt.device_vid, opt.device_pid) if opt.device_vid and opt.device_pid else None,
            fallback_index=index,
        )

        if chosen:
            index = chosen.index
            backend_flag = chosen.backend

            if opt.device_id is None:
                ns = self.settings.properties.setdefault(self.OPTIONS_KEY, {})
                ns["device_id"] = chosen.stable_id
                logger.info("Persisted OpenCV device_id=%s", chosen.stable_id)

        self._capture, spec = open_with_fallbacks(index, backend_flag)

        # 2) Optional Logitech endpoint trick (Windows only)
        # if (
        #     (not self._capture or not self._capture.isOpened())
        #     and platform.system() == "Windows"
        #     and self._alt_index_probe
        # ):
        #     logger.debug("Primary index failed; trying alternate endpoint (index+1) with same backend.")
        #     self._capture = self._try_open(index + 1, backend_flag)

        if not self._capture or not self._capture.isOpened():
            raise RuntimeError(
                f"Unable to open camera index {self.settings.index} with OpenCV (backend {backend_flag})"
            )

        # MSMF hint for slow systems
        if platform.system() == "Windows" and backend_flag == getattr(cv2, "CAP_MSMF", cv2.CAP_ANY):
            if os.environ.get("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS") is None:
                logger.debug(
                    "MSMF selected. If open is slow, consider setting "
                    "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0 before importing cv2."
                )

        if platform.system() == "Windows" and opt.prefer_mjpg and not self._mjpg_attempted:
            self._maybe_enable_mjpg()

        self._configure_capture()

    def read(self) -> tuple[np.ndarray | None, float]:
        """Robust frame read: return (None, ts) on transient failures; never raises."""
        if self._capture is None:
            logger.warning("OpenCVCameraBackend.read() called before open()")
            return None, time.time()
        try:
            if not self._capture.grab():
                return None, time.time()
            success, frame = self._capture.retrieve()
            if not success or frame is None or frame.size == 0:
                return None, time.time()
            return frame, time.time()
        except Exception as exc:
            logger.debug(f"OpenCV read transient error: {exc}")
            return None, time.time()

    def close(self) -> None:
        self._release_capture()

    def stop(self) -> None:
        self._release_capture()

    def device_name(self) -> str:
        base_name = "OpenCV"
        if self._capture and hasattr(self._capture, "getBackendName"):
            try:
                backend_name = self._capture.getBackendName()
            except Exception:
                backend_name = ""
            if backend_name:
                base_name = backend_name
        return f"{base_name} camera #{self.settings.index}"

    @property
    def actual_fps(self) -> float | None:
        """Return the actual configured FPS, if known."""
        return self._actual_fps

    @property
    def actual_resolution(self) -> tuple[int, int] | None:
        """Return the actual configured resolution, if known."""
        if self._actual_width and self._actual_height:
            return (self._actual_width, self._actual_height)
        return None

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _release_capture(self) -> None:
        if self._capture:
            try:
                self._capture.release()
            except Exception:
                pass
            finally:
                self._capture = None
            time.sleep(0.02 if platform.system() == "Windows" else 0.0)

    def _parse_resolution(self, resolution) -> tuple[int, int]:
        if resolution is None:
            return (720, 540)
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            try:
                return (int(resolution[0]), int(resolution[1]))
            except (ValueError, TypeError):
                logger.debug(f"Invalid resolution values: {resolution}, defaulting to 720x540")
                return (720, 540)
        return (720, 540)

    def _preferred_backend_flag(self, backend: str | None) -> int:
        """Resolve preferred backend by platform."""
        if backend:  # user override
            return self._resolve_backend(backend)

        sys = platform.system()
        if sys == "Windows":
            # Prefer DSHOW (faster on many Logitech cams), then MSMF, then ANY.
            return getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
        if sys == "Darwin":
            return getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
        # Linux and others
        return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)

    def _try_open(self, index: int, preferred_flag: int) -> cv2.VideoCapture | None:
        """Try opening with preferred backend, then platform-appropriate fallbacks."""
        # 1) preferred
        cap = cv2.VideoCapture(index, preferred_flag)
        if cap.isOpened():
            return cap

        sys = platform.system()

        # Windows: try MSMF then ANY
        if sys == "Windows":
            ms = getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)
            if preferred_flag != ms:
                cap = cv2.VideoCapture(index, ms)
                if cap.isOpened():
                    return cap

        # macOS: ANY fallback
        if sys == "Darwin":
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
            if cap.isOpened():
                return cap

        # Linux: try ANY as final fallback
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if cap.isOpened():
            return cap
        return None

    def _configure_capture(self) -> None:
        if not self._capture:
            return

        # --- FOURCC (Windows benefits from setting this first) ---
        self._codec_str = self._read_codec_string()
        logger.info(f"Camera using codec: {self._codec_str}")

        # --- Resolution ---
        req_w, req_h = self._resolution
        enforce_aspect = self.parse_options(self.settings).enforce_aspect

        if not self._fast_start:
            result = apply_mode_with_verification(
                self._capture,
                ModeRequest(
                    width=req_w, height=req_h, fps=float(self.settings.fps or 0.0), enforce_aspect=enforce_aspect
                ),
            )
            self._actual_width, self._actual_height, self._actual_fps = result.width, result.height, result.fps
        else:
            self._actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self._actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        # if self._actual_width and self._actual_height:
        #     self.settings.properties["resolution"] = (self._actual_width, self._actual_height)

        # Handle mismatch quickly with a few known-good UVC fallbacks (Windows only)
        if platform.system() == "Windows" and self._actual_width and self._actual_height:
            if (self._actual_width, self._actual_height) != (req_w, req_h) and not self._fast_start:
                logger.warning(
                    f"Resolution mismatch: requested {req_w}x{req_h}, got {self._actual_width}x{self._actual_height}"
                )
                self._resolution = (self._actual_width or req_w, self._actual_height or req_h)
        else:
            # Non-Windows: accept actual as-is
            self._resolution = (self._actual_width or req_w, self._actual_height or req_h)

        logger.info(f"Camera configured with resolution: {self._resolution[0]}x{self._resolution[1]}")

        # --- FPS ---
        requested_fps = float(self.settings.fps or 0.0)
        if not self._fast_start and requested_fps > 0.0:
            current_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if current_fps <= 0.0 or abs(current_fps - requested_fps) > 0.1:
                if not self._capture.set(cv2.CAP_PROP_FPS, requested_fps):
                    logger.debug(f"Device ignored FPS set to {requested_fps:.2f}")
            self._actual_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        else:
            self._actual_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)

        # Log any mismatch
        if self._actual_fps and requested_fps and abs(self._actual_fps - requested_fps) > 0.1:
            logger.warning(f"FPS mismatch: requested {requested_fps:.2f}, got {self._actual_fps:.2f}")

        # Always reconcile the settings with what we measured/obtained
        # if self._actual_fps:
        #     self.settings.fps = float(self._actual_fps)
        logger.info(f"Camera configured with FPS: {self._actual_fps:.2f}")
        logger.debug(
            "CAP_PROP_FPS requested=%s set_ok=%s get=%s",
            self.settings.fps,
            self._capture.set(cv2.CAP_PROP_FPS, float(self.settings.fps)),
            self._capture.get(cv2.CAP_PROP_FPS),
        )

        # --- Extra properties (safe whitelist) ---
        for prop, value in self.settings.properties.items():
            if prop in ("api", "resolution", "fast_start", "alt_index_probe"):
                continue
            try:
                prop_id = int(prop)
            except (TypeError, ValueError):
                logger.debug(f"Ignoring non-numeric property ID: {prop}")
                continue
            if prop_id not in self.SAFE_PROP_IDS:
                logger.debug(f"Skipping unsupported/unsafe property {prop_id}")
                continue
            try:
                if not self._capture.set(prop_id, float(value)):
                    logger.debug(f"Device ignored property {prop_id} -> {value}")
            except Exception as exc:
                logger.debug(f"Failed to set property {prop_id} -> {value}: {exc}")

    # ----------------------------
    # Lower-level helpers
    # ----------------------------

    def _read_codec_string(self) -> str:
        try:
            fourcc = int(self._capture.get(cv2.CAP_PROP_FOURCC) or 0)
        except Exception:
            fourcc = 0
        if fourcc <= 0:
            return ""
        return "".join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)])

    def _maybe_enable_mjpg(self) -> None:
        """Attempt to enable MJPG on Windows devices; verify once."""
        if platform.system() != "Windows":
            return
        try:
            fourcc_mjpg = cv2.VideoWriter_fourcc(*"MJPG")
            if self._capture.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg):
                verify = self._read_codec_string()
                if verify and verify.upper().startswith("MJPG"):
                    logger.info("MJPG enabled successfully.")
                else:
                    logger.debug(f"MJPG set reported success, but codec is '{verify}'")
            else:
                logger.debug("Device rejected MJPG FourCC set.")
        except Exception as exc:
            logger.debug(f"MJPG enable attempt raised: {exc}")

    # def _set_resolution_if_needed(self, width: int, height: int, reconfigure_only: bool = False) -> bool:
    #     """Set width/height only if different.
    #     Returns True if the device ends up at the requested size.
    #     """
    #     try:
    #         cur_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    #         cur_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    #     except Exception:
    #         cur_w, cur_h = 0, 0

    #     if (cur_w != width) or (cur_h != height):
    #         set_w_ok = self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    #         set_h_ok = self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    #         if not set_w_ok:
    #             logger.debug(f"Failed to set frame width to {width}")
    #         if not set_h_ok:
    #             logger.debug(f"Failed to set frame height to {height}")

    #     try:
    #         self._actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    #         self._actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    #     except Exception:
    #         self._actual_width, self._actual_height = 0, 0

    #     return (self._actual_width, self._actual_height) == (width, height)

    def _resolve_backend(self, backend: str | None) -> int:
        if backend is None:
            return cv2.CAP_ANY
        key = backend.upper()
        return getattr(cv2, f"CAP_{key}", cv2.CAP_ANY)

    # ----------------------------
    # Discovery helper (optional use by factory)
    # ----------------------------
    @staticmethod
    def quick_ping(index: int, backend_flag: int | None = None) -> bool:
        """Cheap 'is-present' check to avoid expensive blind opens during discovery."""
        sys = platform.system()
        if sys == "Linux":
            # /dev/videoN present? That's a cheap, reliable hint.
            return os.path.exists(f"/dev/video{index}")
        if backend_flag is None:
            if sys == "Windows":
                backend_flag = getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
            elif sys == "Darwin":
                backend_flag = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
            else:
                backend_flag = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        cap = cv2.VideoCapture(index, backend_flag)
        ok = cap.isOpened()
        try:
            cap.release()
        except Exception:
            pass
        return ok
