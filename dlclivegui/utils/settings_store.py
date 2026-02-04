# dlclivegui/utils/settings_store.py
from pathlib import Path

from PySide6.QtCore import QSettings

from ..config import ApplicationSettings
from .utils import is_model_file


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
            return None


class ModelPathStore:
    """Persist and resolve the last model path via QSettings."""

    def __init__(self, settings: QSettings | None = None):
        self._settings = settings or QSettings("DeepLabCut", "DLCLiveGUI")

    def _norm(self, p: str | None) -> str | None:
        if not p:
            return None
        try:
            return str(Path(p).expanduser())
        except Exception:
            return None

    def load_last(self) -> str | None:
        val = self._settings.value("dlc/last_model_path")
        path = self._norm(str(val)) if val else None
        if not path:
            return None
        try:
            return path if is_model_file(path) else None
        except Exception:
            return None

    def load_last_dir(self) -> str | None:
        val = self._settings.value("dlc/last_model_dir")
        d = self._norm(str(val)) if val else None
        if not d:
            return None
        try:
            p = Path(d)
            return str(p) if p.exists() and p.is_dir() else None
        except Exception:
            return None

    def save_if_valid(self, path: str) -> None:
        """Save last model *file* if it looks valid, and always save its directory."""
        path = self._norm(path) or ""
        if not path:
            return
        try:
            parent = str(Path(path).parent)
            self._settings.setValue("dlc/last_model_dir", parent)

            if is_model_file(path):
                self._settings.setValue("dlc/last_model_path", str(Path(path)))
        except Exception:
            pass

    def save_last_dir(self, directory: str) -> None:
        directory = self._norm(directory) or ""
        if not directory:
            return
        try:
            p = Path(directory)
            if p.exists() and p.is_dir():
                self._settings.setValue("dlc/last_model_dir", str(p))
        except Exception:
            pass

    def resolve(self, config_path: str | None) -> str:
        """Resolve the best model path to display in the UI."""
        config_path = self._norm(config_path)
        if config_path:
            try:
                if is_model_file(config_path):
                    return config_path
            except Exception:
                pass

        persisted = self.load_last()
        if persisted:
            try:
                if is_model_file(persisted):
                    return persisted
            except Exception:
                pass

        return ""

    def suggest_start_dir(self, fallback_dir: str | None = None) -> str:
        """Pick the best directory to start the file dialog in."""
        # 1) last dir
        last_dir = self.load_last_dir()
        if last_dir:
            return last_dir

        # 2) directory of last valid model file
        last_file = self.load_last()
        if last_file:
            try:
                parent = Path(last_file).parent
                if parent.exists():
                    return str(parent)
            except Exception:
                pass

        # 3) fallback dir (config.model_directory) if valid
        if fallback_dir:
            try:
                p = Path(fallback_dir).expanduser()
                if p.exists() and p.is_dir():
                    return str(p)
            except Exception:
                pass

        # 4) last resort: home
        return str(Path.home())

    def suggest_selected_file(self) -> str | None:
        """Optional: return a file to preselect if it exists."""
        last_file = self.load_last()
        if not last_file:
            return None
        try:
            p = Path(last_file)
            return str(p) if p.exists() and p.is_file() else None
        except Exception:
            return None
