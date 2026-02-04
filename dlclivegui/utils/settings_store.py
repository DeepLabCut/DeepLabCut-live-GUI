# dlclivegui/utils/settings_store.py
from PySide6.QtCore import QSettings

from .config_models import ApplicationSettingsModel


class QtSettingsStore:
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

    # --- optional: snapshot full config as JSON in QSettings ---
    def save_full_config_snapshot(self, cfg: ApplicationSettingsModel) -> None:
        self._s.setValue("app/config_json", cfg.model_dump_json())

    def load_full_config_snapshot(self) -> ApplicationSettingsModel | None:
        raw = self._s.value("app/config_json", "")
        if not raw:
            return None
        try:
            return ApplicationSettingsModel.model_validate_json(str(raw))
        except Exception:
            return None
