# dlclivegui/utils/settings_store.py
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QSettings

from ..config import ApplicationSettings
from .utils import is_model_file

logger = logging.getLogger(__name__)


class DLCLiveGUISettingsStore:
    def __init__(self, qsettings: QSettings | None = None):
        self._s = qsettings or QSettings("DeepLabCut", "DLCLiveGUI")

    # --- lightweight prefs ---
    def get_last_model_path(self) -> str | None:
        v = self._s.value("dlc/last_model_path", "")
        return str(v) if v else None

    def set_last_model_path(self, path: str) -> None:
        self._s.setValue("dlc/last_model_path", path or "")

    def get_last_config_path(self) -> str | None:
        v = self._s.value("app/last_config_path", "")
        return str(v) if v else None

    def set_last_config_path(self, path: str) -> None:
        self._s.setValue("app/last_config_path", path or "")

    def get_session_name(self) -> str:
        v = self._s.value("recording/session_name", "")
        return str(v) if v else ""

    def set_session_name(self, name: str) -> None:
        self._s.setValue("recording/session_name", name or "")

    def get_use_timestamp(self, default: bool = True) -> bool:
        v = self._s.value("recording/use_timestamp", default)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(default)

    def set_use_timestamp(self, value: bool) -> None:
        self._s.setValue("recording/use_timestamp", bool(value))

    # --- optional: snapshot full config as JSON in QSettings ---
    def save_full_config_snapshot(self, cfg: ApplicationSettings) -> None:
        self._s.setValue("app/config_json", cfg.model_dump_json())

    def load_full_config_snapshot(self) -> ApplicationSettings | None:
        raw = self._s.value("app/config_json", "")
        if not raw:
            return None
        try:
            return ApplicationSettings.model_validate_json(str(raw))
        except Exception:
            logger.debug("Failed to load full config snapshot from QSettings")
            return None


class ModelPathStore:
    """Persist and resolve the last model path via QSettings."""

    def __init__(self, settings: QSettings | None = None):
        self._settings = settings or QSettings("DeepLabCut", "DLCLiveGUI")

    # -------------------------
    # Normalization helpers
    # -------------------------
    def _as_path(self, p: str | None) -> Path | None:
        """Best-effort conversion to Path (expand ~, interpret '.' as cwd)."""
        if not p:
            return None
        s = str(p).strip()
        if not s:
            return None
        try:
            pp = Path(s).expanduser()
            if s in (".", "./"):
                pp = Path.cwd()
            return pp
        except Exception:
            logger.debug("Failed to parse path: %s", p)
            return None

    def _norm_existing_dir(self, p: str | None) -> str | None:
        """Return an absolute, resolved existing directory path, else None."""
        pp = self._as_path(p)
        if pp is None:
            return None
        try:
            # If a file was given, use its parent directory
            if pp.exists() and pp.is_file():
                pp = pp.parent

            if pp.exists() and pp.is_dir():
                return str(pp.resolve())
        except Exception:
            logger.debug("Failed to normalize directory: %s", p)
        return None

    def _norm_existing_path(self, p: str | None) -> str | None:
        """Return an absolute, resolved existing path (file or dir), else None."""
        pp = self._as_path(p)
        if pp is None:
            return None
        try:
            if pp.exists():
                return str(pp.resolve())
        except Exception:
            logger.debug("Failed to normalize path: %s", p)
        return None

    # -------------------------
    # Load
    # -------------------------
    def load_last(self) -> str | None:
        """Return last model path if it still exists and looks usable."""
        val = self._settings.value("dlc/last_model_path")
        path = self._norm_existing_path(str(val)) if val else None
        if not path:
            return None

        try:
            pp = Path(path)
            # Accept a valid model *file*
            if pp.is_file() and is_model_file(str(pp)):
                return str(pp)
        except Exception:
            logger.debug("Last model path not valid/usable: %s", path)

        return None

    def load_last_dir(self) -> str | None:
        """Return last directory if it still exists and is a directory."""
        val = self._settings.value("dlc/last_model_dir")
        d = self._norm_existing_dir(str(val)) if val else None
        return d

    # -------------------------
    # Save
    # -------------------------
    def save_if_valid(self, path: str) -> None:
        """
        Save last model path if it looks valid/usable, and always save its directory.
        - For files: always save parent directory.
        - For directories: save directory itself if it looks like a TF model dir.
        """
        norm = self._norm_existing_path(path)
        if not norm:
            return

        try:
            p = Path(norm)

            # Always persist a *directory* that is safe for QFileDialog.setDirectory(...)
            if p.is_dir():
                model_dir = p
            else:
                model_dir = p.parent

            model_dir_norm = self._norm_existing_dir(str(model_dir))
            if model_dir_norm:
                self._settings.setValue("dlc/last_model_dir", model_dir_norm)

            # Persist model path if it is a valid model file, or a TF model directory
            if p.is_file() and is_model_file(str(p)):
                self._settings.setValue("dlc/last_model_path", str(p))
            elif p.is_dir() and self._looks_like_tf_model_dir(p):
                self._settings.setValue("dlc/last_model_path", str(p))

        except Exception:
            logger.debug("Failed to save model path: %s", path, exc_info=True)

    def save_last_dir(self, directory: str) -> None:
        d = self._norm_existing_dir(directory)
        if not d:
            return
        try:
            self._settings.setValue("dlc/last_model_dir", d)
        except Exception:
            logger.debug("Failed to save last model dir: %s", d, exc_info=True)

    # -------------------------
    # Resolve
    # -------------------------
    def resolve(self, config_path: str | None) -> str:
        """
        Resolve the best model path to display in the UI.
        Preference:
          1) config_path if valid/usable
          2) persisted last model path if valid/usable
          3) empty
        """
        cfg = self._norm_existing_path(config_path)
        if cfg:
            try:
                p = Path(cfg)
                if p.is_file() and is_model_file(cfg):
                    return cfg
                if p.is_dir() and self._looks_like_tf_model_dir(p):
                    return cfg
            except Exception:
                logger.debug("Config path not usable: %s", cfg)

        persisted = self.load_last()
        if persisted:
            return persisted

        return ""

    def suggest_start_dir(self, fallback_dir: str | None = None) -> str:
        """
        Pick the best directory to start file dialogs in.
        Guarantees: returns an existing absolute directory (never '.').
        """
        # 1) last dir
        last_dir = self.load_last_dir()
        if last_dir:
            return last_dir

        # 2) directory of last valid model path
        last = self.load_last()
        if last:
            try:
                p = Path(last)
                if p.is_file():
                    parent = self._norm_existing_dir(str(p.parent))
                    if parent:
                        return parent
                elif p.is_dir():
                    d = self._norm_existing_dir(str(p))
                    if d:
                        return d
            except Exception:
                logger.debug("Failed to derive start dir from last model: %s", last)

        # 3) fallback dir (e.g. config.dlc.model_directory)
        fb = self._norm_existing_dir(fallback_dir)
        if fb:
            return fb

        # 4) last resort: cwd if exists else home
        cwd = self._norm_existing_dir(str(Path.cwd()))
        return cwd or str(Path.home())

    def suggest_selected_file(self) -> str | None:
        """Return a file to preselect if it exists (only files, not directories)."""
        last = self.load_last()
        if not last:
            return None
        try:
            p = Path(last)
            return str(p) if p.exists() and p.is_file() else None
        except Exception:
            logger.debug("Failed to check existence of last model: %s", last)
            return None
