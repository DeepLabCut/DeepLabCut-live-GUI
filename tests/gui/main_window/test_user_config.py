# tests/gui/main_window/test_user_config.py
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.gui
class TestUserConfigPersistence:
    def test_valid_config_file_path_accepts_existing_file(self, window, tmp_path: Path):
        w = window

        config_path = tmp_path / "dlclive_config.json"
        config_path.write_text("{}", encoding="utf-8")

        assert w._valid_config_file_path(str(config_path)) == config_path.resolve()

    def test_valid_config_file_path_rejects_missing_file(self, window, tmp_path: Path):
        w = window

        missing = tmp_path / "missing_config.json"

        assert w._valid_config_file_path(str(missing)) is None
        assert w._valid_config_file_path(None) is None
        assert w._valid_config_file_path("") is None

    def test_suggest_config_dialog_path_prefers_current_config_path(self, window, tmp_path: Path):
        w = window

        config_path = tmp_path / "current_config.json"
        config_path.write_text("{}", encoding="utf-8")

        w._config_path = config_path

        assert w._suggest_config_dialog_path() == str(config_path)

    def test_suggest_config_dialog_path_uses_last_config_path_when_current_path_missing(
        self,
        monkeypatch,
        window,
        tmp_path: Path,
    ):
        w = window

        config_path = tmp_path / "last_config.json"
        config_path.write_text("{}", encoding="utf-8")

        w._config_path = None
        monkeypatch.setattr(w._settings_store, "get_last_config_path", lambda: str(config_path))

        assert w._suggest_config_dialog_path() == str(config_path.resolve())

    def test_suggest_config_dialog_path_uses_parent_of_missing_last_config(
        self,
        monkeypatch,
        window,
        tmp_path: Path,
    ):
        w = window

        missing_config_path = tmp_path / "missing_config.json"

        w._config_path = None
        monkeypatch.setattr(w._settings_store, "get_last_config_path", lambda: str(missing_config_path))

        assert w._suggest_config_dialog_path() == str(missing_config_path)

    def test_save_config_to_path_persists_last_path_snapshot_and_syncs(
        self,
        monkeypatch,
        window,
        tmp_path: Path,
    ):
        w = window

        calls: list[tuple[str, object]] = []

        class FakeConfig:
            def save(self, path: Path | str) -> None:
                Path(path).write_text("{}", encoding="utf-8")

        class FakeSettings:
            def sync(self) -> None:
                calls.append(("sync", True))

        config_path = tmp_path / "saved_config.json"
        fake_config = FakeConfig()

        monkeypatch.setattr(w, "_current_config", lambda allow_empty_model_path=False: fake_config)
        monkeypatch.setattr(
            w._settings_store,
            "set_last_config_path",
            lambda path: calls.append(("last_path", path)),
        )
        monkeypatch.setattr(
            w._settings_store,
            "save_full_config_snapshot",
            lambda cfg: calls.append(("snapshot", cfg)),
        )
        monkeypatch.setattr(w, "settings", FakeSettings())

        assert w._save_config_to_path(config_path) is True
        assert config_path.exists()
        assert ("last_path", str(config_path.resolve())) in calls
        assert ("snapshot", fake_config) in calls
        assert ("sync", True) in calls

    def test_save_config_to_path_returns_false_without_persisting_after_failure(
        self,
        monkeypatch,
        window,
        tmp_path: Path,
    ):
        w = window

        calls: list[tuple[str, object]] = []
        errors: list[str] = []

        class FakeConfig:
            def save(self, path: Path | str) -> None:
                raise OSError("cannot save")

        config_path = tmp_path / "failed_config.json"

        monkeypatch.setattr(w, "_current_config", lambda allow_empty_model_path=False: FakeConfig())
        monkeypatch.setattr(
            w._settings_store,
            "set_last_config_path",
            lambda path: calls.append(("last_path", path)),
        )
        monkeypatch.setattr(
            w._settings_store,
            "save_full_config_snapshot",
            lambda cfg: calls.append(("snapshot", cfg)),
        )
        monkeypatch.setattr(w, "_show_error", errors.append)

        assert w._save_config_to_path(config_path) is False
        assert not config_path.exists()
        assert calls == []
        assert errors
