# dlclivegui/utils/settings_store.py
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QSettings

from ..config import ApplicationSettings
from ..temp import Engine  # type: ignore # TODO use main package enum when released

logger = logging.getLogger(__name__)


class DLCLiveGUISettingsStore:
    """Small QSettings-backed store for lightweight GUI preferences.

    Stores UI/session preferences that should survive
    application restarts but do not necessarily belong in exported JSON configs.

    Full application configuration snapshots are also stored here separately as
    JSON for convenient startup restore.
    """

    # --- app/config keys ---
    KEY_LAST_CONFIG_PATH = "app/last_config_path"
    KEY_CONFIG_JSON = "app/config_json"

    # --- dlc/model keys ---
    KEY_LAST_MODEL_PATH = "dlc/last_model_path"
    KEY_PROCESSOR_FOLDER = "dlc/processor_folder"
    KEY_INFERENCE_CAMERA_ID = "dlc/inference_camera_id"
    KEY_PROCESSOR_KEY = "dlc/processor_key"
    KEY_PROCESSOR_CONTROL_ENABLED = "dlc/processor_control_enabled"

    # --- recording keys ---
    KEY_SESSION_NAME = "recording/session_name"
    KEY_USE_TIMESTAMP = "recording/use_timestamp"
    KEY_FAST_ENCODING = "recording/fast_encoding"

    def __init__(self, qsettings: QSettings | None = None):
        self._s = qsettings or QSettings("DeepLabCut", "DLCLiveGUI")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_bool(self, key: str, default: bool = False) -> bool:
        """Read a bool from QSettings, handling Qt/string/int variants."""
        value = self._s.value(key, default)

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False

        return bool(default)

    def _get_optional_str(self, key: str, default: str = "") -> str | None:
        """Read an optional string from QSettings."""
        value = self._s.value(key, default)
        value = str(value).strip() if value is not None else ""
        return value or None

    def _set_optional_str(self, key: str, value: str | None) -> None:
        """Persist optional string values as empty strings when unset."""
        self._s.setValue(key, str(value).strip() if value else "")

    # ------------------------------------------------------------------
    # App/model prefs
    # ------------------------------------------------------------------
    def get_last_model_path(self) -> str | None:
        return self._get_optional_str(self.KEY_LAST_MODEL_PATH)

    def set_last_model_path(self, path: str) -> None:
        self._set_optional_str(self.KEY_LAST_MODEL_PATH, path)

    def get_last_config_path(self) -> str | None:
        return self._get_optional_str(self.KEY_LAST_CONFIG_PATH)

    def set_last_config_path(self, path: str) -> None:
        self._set_optional_str(self.KEY_LAST_CONFIG_PATH, path)

    # ------------------------------------------------------------------
    # Recording prefs
    # ------------------------------------------------------------------
    def get_session_name(self) -> str:
        return self._get_optional_str(self.KEY_SESSION_NAME) or ""

    def set_session_name(self, name: str) -> None:
        self._set_optional_str(self.KEY_SESSION_NAME, name)

    def get_use_timestamp(self, default: bool = True) -> bool:
        return self._get_bool(self.KEY_USE_TIMESTAMP, default=default)

    def set_use_timestamp(self, value: bool) -> None:
        self._s.setValue(self.KEY_USE_TIMESTAMP, bool(value))

    def get_fast_encoding(self, default: bool = False) -> bool:
        return self._get_bool(self.KEY_FAST_ENCODING, default=default)

    def set_fast_encoding(self, enabled: bool) -> None:
        self._s.setValue(self.KEY_FAST_ENCODING, bool(enabled))

    # ------------------------------------------------------------------
    # DLC camera / processor prefs
    # ------------------------------------------------------------------
    def get_inference_camera_id(self, default: str | None = None) -> str | None:
        """Return the last explicitly selected DLC inference camera ID.

        This is a user preference. Runtime fallbacks during preview should not
        overwrite this value unless the user explicitly changes the combo.
        """
        return self._get_optional_str(self.KEY_INFERENCE_CAMERA_ID, default or "")

    def set_inference_camera_id(self, camera_id: str | None) -> None:
        """Persist the explicitly selected DLC inference camera ID."""
        self._set_optional_str(self.KEY_INFERENCE_CAMERA_ID, camera_id)

    def get_processor_key(self, default: str | None = None) -> str | None:
        """Return the last selected processor key, if any."""
        return self._get_optional_str(self.KEY_PROCESSOR_KEY, default or "")

    def set_processor_key(self, processor_key: str | None) -> None:
        """Persist the selected processor key.

        The key may become unavailable if the processor folder changes. In that
        case the GUI should simply fall back to "No Processor" while keeping
        refresh behavior graceful.
        """
        self._set_optional_str(self.KEY_PROCESSOR_KEY, processor_key)

    def get_processor_control_enabled(self, default: bool = False) -> bool:
        """Return whether processor-based control was enabled last time."""
        return self._get_bool(self.KEY_PROCESSOR_CONTROL_ENABLED, default=default)

    def set_processor_control_enabled(self, enabled: bool) -> None:
        """Persist processor-based control checkbox state."""
        self._s.setValue(self.KEY_PROCESSOR_CONTROL_ENABLED, bool(enabled))

    def get_processor_folder(self, default: str = "") -> str:
        """Return the persisted processor folder if it still exists.

        If the stored folder is missing or invalid, return default.
        """
        value = self._s.value(self.KEY_PROCESSOR_FOLDER, default)
        value = str(value).strip() if value is not None else ""

        if not value:
            return default

        try:
            path = Path(value).expanduser()
            if path.is_dir():
                return str(path.resolve())
        except Exception:
            logger.debug("Persisted processor folder is invalid: %s", value, exc_info=True)

        return default

    def set_processor_folder(self, folder: str) -> None:
        """Persist processor folder only if it exists and is a directory.

        Invalid folders are ignored so we do not accidentally replace a valid
        stored folder with an unusable value.
        """
        folder = str(folder).strip() if folder is not None else ""
        if not folder:
            return

        try:
            path = Path(folder).expanduser()
            if path.is_dir():
                self._s.setValue(self.KEY_PROCESSOR_FOLDER, str(path.resolve()))
        except Exception:
            logger.debug("Failed to persist processor folder: %s", folder, exc_info=True)

    # ------------------------------------------------------------------
    # Full config snapshot
    # ------------------------------------------------------------------
    def save_full_config_snapshot(self, cfg: ApplicationSettings) -> None:
        """Persist the current full application config as JSON in QSettings."""
        self._s.setValue(self.KEY_CONFIG_JSON, cfg.model_dump_json())

    def load_full_config_snapshot(self) -> ApplicationSettings | None:
        """Load the previously persisted full application config snapshot."""
        raw = self._s.value(self.KEY_CONFIG_JSON, "")
        if not raw:
            return None

        try:
            return ApplicationSettings.model_validate_json(str(raw))
        except Exception:
            logger.debug("Failed to load full config snapshot from QSettings", exc_info=True)
            return None


class ModelPathStore:
    """Persist and resolve the last model path via QSettings."""

    def __init__(self, settings: QSettings | None = None):
        self._settings = settings or QSettings("DeepLabCut", "DLCLiveGUI")

    # -------------------------
    # Normalization helpers
    # -------------------------
    def _as_path(self, p: str | None) -> Path | None:
        """Best-effort conversion to Path.

        Expands '~' and interprets '.' as the current working directory.
        """
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
            logger.debug("Failed to parse path: %s", p, exc_info=True)
            return None

    def _norm_existing_dir(self, p: str | None) -> str | None:
        """Return an absolute, resolved existing directory path, else None."""
        pp = self._as_path(p)
        if pp is None:
            return None

        try:
            # If a file was given, use its parent directory.
            if pp.exists() and pp.is_file():
                pp = pp.parent

            if pp.exists() and pp.is_dir():
                return str(pp.resolve())
        except Exception:
            logger.debug("Failed to normalize directory: %s", p, exc_info=True)

        return None

    def _norm_existing_path(self, p: str | None) -> str | None:
        """Return an absolute, resolved existing path, file or dir, else None."""
        pp = self._as_path(p)
        if pp is None:
            return None

        try:
            if pp.exists():
                return str(pp.resolve())
        except Exception:
            logger.debug("Failed to normalize path: %s", p, exc_info=True)

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

            # Accept a valid model file.
            if pp.is_file() and (Engine.is_pytorch_model_path(pp) or Engine.is_tensorflow_model_dir_path(pp.parent)):
                return str(pp)
        except Exception:
            logger.debug("Last model path not valid/usable: %s", path, exc_info=True)

        return None

    def load_last_dir(self) -> str | None:
        """Return last directory if it still exists and is a directory."""
        val = self._settings.value("dlc/last_model_dir")
        return self._norm_existing_dir(str(val)) if val else None

    # -------------------------
    # Save
    # -------------------------
    def save_if_valid(self, path: str) -> None:
        """Save last model path if it looks valid/usable.

        Also saves a safe directory for QFileDialog.setDirectory(...).

        - For files: saves parent directory.
        - For directories: saves the directory itself when appropriate.
        """
        norm = self._norm_existing_path(path)
        if not norm:
            return

        try:
            p = Path(norm)

            # Always persist a directory that is safe for QFileDialog.setDirectory(...).
            if p.is_dir():
                model_dir = p
            else:
                model_dir = p.parent

            model_dir_norm = self._norm_existing_dir(str(model_dir))
            if model_dir_norm:
                self._settings.setValue("dlc/last_model_dir", model_dir_norm)

            # Persist model path if it is a valid model file, or a TF model file
            # whose parent is a TensorFlow model directory.
            if Engine.is_pytorch_model_path(p):
                self._settings.setValue("dlc/last_model_path", str(p))
            elif p.parent.is_dir() and Engine.is_tensorflow_model_dir_path(p.parent):
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
        """Resolve the best model path to display in the UI.

        Preference:
          1. config_path if valid/usable
          2. persisted last model path if valid/usable
          3. empty string
        """
        cfg = self._norm_existing_path(config_path)
        if cfg:
            try:
                p = Path(cfg)
                if p.is_file() and Engine.is_pytorch_model_path(p):
                    return cfg
                if p.is_dir() and Engine.is_tensorflow_model_dir_path(p):
                    return cfg
            except Exception:
                logger.debug("Config path not usable: %s", cfg, exc_info=True)

        persisted = self.load_last()
        if persisted:
            return persisted

        return ""

    def suggest_start_dir(self, fallback_dir: str | None = None) -> str:
        """Pick the best directory to start file dialogs in.

        Guarantees: returns an existing absolute directory, never '.'.
        """
        # 1. last dir
        last_dir = self.load_last_dir()
        if last_dir:
            return last_dir

        # 2. directory of last valid model path
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
                logger.debug("Failed to derive start dir from last model: %s", last, exc_info=True)

        # 3. fallback dir, e.g. config.dlc.model_directory
        fb = self._norm_existing_dir(fallback_dir)
        if fb:
            return fb

        # 4. last resort: cwd if exists else home
        cwd = self._norm_existing_dir(str(Path.cwd()))
        return cwd or str(Path.home())

    def suggest_selected_file(self) -> str | None:
        """Return a file to preselect if it exists.

        Only files are returned, not directories.
        """
        last = self.load_last()
        if not last:
            return None

        try:
            p = Path(last)
            return str(p) if p.exists() and p.is_file() else None
        except Exception:
            logger.debug("Failed to check existence of last model: %s", last, exc_info=True)
            return None
