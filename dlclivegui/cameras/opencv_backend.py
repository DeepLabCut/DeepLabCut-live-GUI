"""OpenCV-based camera backend (Windows-optimized, fast startup, robust read)."""

from __future__ import annotations

import logging
import platform
import time

import cv2
import numpy as np

from .base import CameraBackend

LOG = logging.getLogger(__name__)


class OpenCVCameraBackend(CameraBackend):
    """Backend using :mod:`cv2.VideoCapture` with Windows/MSMF preference and safe MJPG attempt.

    Key features:
    - Prefers MediaFoundation (MSMF) on Windows; falls back to DirectShow (DSHOW) or ANY.
    - Attempts to enable MJPG **only** on Windows and **only** if the device accepts it.
    - Minimizes expensive property negotiations (width/height/FPS) to what’s really needed.
    - Robust `read()` that returns (None, ts) on transient failures instead of raising.
    - Optional fast-start mode: set `properties["fast_start"]=True` to skip noncritical sets.
    """

    # Whitelisted camera properties we allow from settings.properties (numeric IDs only).
    SAFE_PROP_IDS = {
        # Exposure: note Windows backends differ in support (some expect relative values)
        int(getattr(cv2, "CAP_PROP_EXPOSURE", 15)),
        int(getattr(cv2, "CAP_PROP_AUTO_EXPOSURE", 21)),
        # Gain (not always supported)
        int(getattr(cv2, "CAP_PROP_GAIN", 14)),
        # FPS (read-only on many webcams; we still attempt)
        int(getattr(cv2, "CAP_PROP_FPS", 5)),
        # Brightness / Contrast (optional, many cams support)
        int(getattr(cv2, "CAP_PROP_BRIGHTNESS", 10)),
        int(getattr(cv2, "CAP_PROP_CONTRAST", 11)),
        int(getattr(cv2, "CAP_PROP_SATURATION", 12)),
        int(getattr(cv2, "CAP_PROP_HUE", 13)),
        # Disable RGB conversion (can reduce overhead if needed)
        int(getattr(cv2, "CAP_PROP_CONVERT_RGB", 17)),
    }

    def __init__(self, settings):
        super().__init__(settings)
        self._capture: cv2.VideoCapture | None = None
        self._resolution: tuple[int, int] = self._parse_resolution(settings.properties.get("resolution"))
        # Optional fast-start: skip some property sets to reduce startup latency.
        self._fast_start: bool = bool(self.settings.properties.get("fast_start", False))
        # Cache last-known device state to avoid repeated queries
        self._actual_width: int | None = None
        self._actual_height: int | None = None
        self._actual_fps: float | None = None
        self._codec_str: str = ""

    # ----------------------------
    # Public API
    # ----------------------------

    def open(self) -> None:
        backend_flag = self._preferred_backend_flag(self.settings.properties.get("api"))
        index = int(self.settings.index)

        # Try preferred backend, then fallback chain
        self._capture = self._try_open(index, backend_flag)
        if not self._capture or not self._capture.isOpened():
            raise RuntimeError(
                f"Unable to open camera index {self.settings.index} with OpenCV (backend {backend_flag})"
            )

        self._configure_capture()

    def read(self) -> tuple[np.ndarray | None, float]:
        """Robust frame read: return (None, ts) on transient failures; never raises."""
        if self._capture is None:
            # This should never happen in normal operation.
            LOG.warning("OpenCVCameraBackend.read() called before open()")
            return None, time.time()

        # Some Windows webcams intermittently fail grab/retrieve.
        # We *do not* raise, to avoid GUI restarts / loops.
        try:
            if not self._capture.grab():
                return None, time.time()
            success, frame = self._capture.retrieve()
            if not success or frame is None or frame.size == 0:
                return None, time.time()
            return frame, time.time()
        except Exception as exc:
            # Log at debug to avoid warning spam
            LOG.debug(f"OpenCV read transient error: {exc}")
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
            # Small pause helps certain Windows drivers settle after release.
            time.sleep(0.02 if platform.system() == "Windows" else 0.0)

    def _parse_resolution(self, resolution) -> tuple[int, int]:
        if resolution is None:
            return (720, 540)
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            try:
                return (int(resolution[0]), int(resolution[1]))
            except (ValueError, TypeError):
                LOG.debug(f"Invalid resolution values: {resolution}, defaulting to 720x540")
                return (720, 540)
        return (720, 540)

    def _preferred_backend_flag(self, backend: str | None) -> int:
        """Resolve preferred backend, with Windows-aware defaults."""
        if backend:  # explicit request from settings
            return self._resolve_backend(backend)

        # Default preference by platform:
        if platform.system() == "Windows":
            # Prefer MSMF on modern Windows; fallback to DSHOW if needed.
            return getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)
        else:
            # Non-Windows: let OpenCV pick
            return cv2.CAP_ANY

    def _try_open(self, index: int, preferred_flag: int) -> cv2.VideoCapture | None:
        """Try opening with preferred backend, then fall back."""
        # 1) preferred
        cap = cv2.VideoCapture(index, preferred_flag)
        if cap.isOpened():
            return cap

        # 2) Windows fallback chain
        if platform.system() == "Windows":
            # If preferred was MSMF, try DSHOW, then ANY
            dshow = getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
            if preferred_flag != dshow:
                cap = cv2.VideoCapture(index, dshow)
                if cap.isOpened():
                    return cap

        # 3) Any
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if cap.isOpened():
            return cap

        return None

    def _configure_capture(self) -> None:
        if not self._capture:
            return

        # --- Codec (FourCC) ---
        self._codec_str = self._read_codec_string()
        LOG.info(f"Camera using codec: {self._codec_str}")

        # Attempt MJPG on Windows only, then re-read codec
        if platform.system() == "Windows":
            self._maybe_enable_mjpg()
            self._codec_str = self._read_codec_string()
            LOG.info(f"Camera codec after MJPG attempt: {self._codec_str}")

        # --- Resolution ---
        width, height = self._resolution
        if not self._fast_start:
            self._set_resolution_if_needed(width, height)
        else:
            # Fast-start: Avoid early set; just read actual once for logging.
            self._actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self._actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # If mismatch, update internal state for downstream consumers (avoid retries).
        if self._actual_width and self._actual_height:
            if (self._actual_width != width) or (self._actual_height != height):
                LOG.warning(
                    f"Resolution mismatch: requested {width}x{height}, got {self._actual_width}x{self._actual_height}"
                )
                self._resolution = (self._actual_width, self._actual_height)

        LOG.info(f"Camera configured with resolution: {self._resolution[0]}x{self._resolution[1]}")

        # --- FPS ---
        requested_fps = float(self.settings.fps or 0.0)
        if not self._fast_start and requested_fps > 0.0:
            # Only set if different and meaningful
            current_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if current_fps <= 0.0 or abs(current_fps - requested_fps) > 0.1:
                if not self._capture.set(cv2.CAP_PROP_FPS, requested_fps):
                    LOG.debug(f"Device ignored FPS set to {requested_fps:.2f}")
            # Re-read
            self._actual_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        else:
            # Fast-start: just read for logging
            self._actual_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)

        if self._actual_fps and requested_fps:
            if abs(self._actual_fps - requested_fps) > 0.1:
                LOG.warning(f"FPS mismatch: requested {requested_fps:.2f}, got {self._actual_fps:.2f}")
        if self._actual_fps:
            self.settings.fps = float(self._actual_fps)
            LOG.info(f"Camera configured with FPS: {self._actual_fps:.2f}")

        # --- Extra properties (whitelisted only, numeric IDs only) ---
        for prop, value in self.settings.properties.items():
            if prop in ("api", "resolution", "fast_start"):
                continue
            try:
                prop_id = int(prop)
            except (TypeError, ValueError):
                # Named properties are not supported here; keep numeric only
                LOG.debug(f"Ignoring non-numeric property ID: {prop}")
                continue

            if prop_id not in self.SAFE_PROP_IDS:
                LOG.debug(f"Skipping unsupported/unsafe property {prop_id}")
                continue

            try:
                if not self._capture.set(prop_id, float(value)):
                    LOG.debug(f"Device ignored property {prop_id} -> {value}")
            except Exception as exc:
                LOG.debug(f"Failed to set property {prop_id} -> {value}: {exc}")

    # ----------------------------
    # Lower-level helpers
    # ----------------------------

    def _read_codec_string(self) -> str:
        """Get FourCC as text; returns empty if not available."""
        try:
            fourcc = int(self._capture.get(cv2.CAP_PROP_FOURCC) or 0)
        except Exception:
            fourcc = 0
        if fourcc <= 0:
            return ""
        # FourCC in little-endian order
        return "".join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)])

    def _maybe_enable_mjpg(self) -> None:
        """Attempt to enable MJPG on Windows devices; verify and log."""
        try:
            fourcc_mjpg = cv2.VideoWriter_fourcc(*"MJPG")
            if self._capture.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg):
                # Verify
                verify = self._read_codec_string()
                if verify and verify.upper().startswith("MJPG"):
                    LOG.info("MJPG enabled successfully.")
                else:
                    LOG.debug(f"MJPG set reported success, but codec is '{verify}'")
            else:
                LOG.debug("Device rejected MJPG FourCC set.")
        except Exception as exc:
            LOG.debug(f"MJPG enable attempt raised: {exc}")

    def _set_resolution_if_needed(self, width: int, height: int) -> None:
        """Set width/height only if different to minimize renegotiation cost."""
        # Read current
        try:
            cur_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            cur_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        except Exception:
            cur_w, cur_h = 0, 0

        # Only set if different
        if (cur_w != width) or (cur_h != height):
            # Set desired
            set_w_ok = self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            set_h_ok = self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            if not set_w_ok:
                LOG.debug(f"Failed to set frame width to {width}")
            if not set_h_ok:
                LOG.debug(f"Failed to set frame height to {height}")

        # Re-read actual and cache
        try:
            self._actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self._actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        except Exception:
            self._actual_width, self._actual_height = 0, 0

    def _resolve_backend(self, backend: str | None) -> int:
        if backend is None:
            return cv2.CAP_ANY
        key = backend.upper()
        # Common aliases: MSMF, DSHOW, ANY, V4L2 (non-Windows)
        return getattr(cv2, f"CAP_{key}", cv2.CAP_ANY)
