from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

import dlclivegui.utils.settings_store as store

pytestmark = pytest.mark.unit


class InMemoryQSettings:
    """Stand-in for QSettings"""

    def __init__(self):
        self._d = {}

    def value(self, key: str, default=None):
        return self._d.get(key, default)

    def setValue(self, key: str, value):
        self._d[key] = value


# -----------------------------
# QtSettingsStore
# -----------------------------
def test_qt_settings_store_last_paths_roundtrip():
    s = InMemoryQSettings()
    settstore = store.DLCLiveGUISettingsStore(qsettings=s)

    assert settstore.get_last_model_path() is None
    assert settstore.get_last_config_path() is None

    settstore.set_last_model_path("/tmp/model.pt")
    settstore.set_last_config_path("/tmp/config.yaml")

    assert settstore.get_last_model_path() == "/tmp/model.pt"
    assert settstore.get_last_config_path() == "/tmp/config.yaml"

    # Empty strings should come back as None
    settstore.set_last_model_path("")
    settstore.set_last_config_path("")
    assert settstore.get_last_model_path() is None
    assert settstore.get_last_config_path() is None


def test_qt_settings_store_full_config_snapshot_ok(monkeypatch):
    s = InMemoryQSettings()
    settstore = store.DLCLiveGUISettingsStore(qsettings=s)

    @dataclass
    class FakeAppSettings:
        x: int = 1

        def model_dump_json(self) -> str:
            return '{"x": 1}'

        @staticmethod
        def model_validate_json(raw: str):
            # Return a recognizable object
            return FakeAppSettings(x=1)

    # Patch the imported symbol in the module under test
    monkeypatch.setattr(store, "ApplicationSettings", FakeAppSettings)

    cfg = FakeAppSettings(x=1)
    settstore.save_full_config_snapshot(cfg)

    loaded = settstore.load_full_config_snapshot()
    assert isinstance(loaded, FakeAppSettings)
    assert loaded.x == 1


def test_qt_settings_store_full_config_snapshot_invalid_returns_none(monkeypatch):
    s = InMemoryQSettings()
    settstore = store.DLCLiveGUISettingsStore(qsettings=s)

    @dataclass
    class FakeAppSettings:
        x: int = 1

        def model_dump_json(self) -> str:
            return "NOT JSON"

        @staticmethod
        def model_validate_json(raw: str):
            raise ValueError("bad json")

    monkeypatch.setattr(store, "ApplicationSettings", FakeAppSettings)

    # store invalid json
    settstore.save_full_config_snapshot(FakeAppSettings(x=1))
    assert settstore.load_full_config_snapshot() is None


# -----------------------------
# ModelPathStore helpers
# -----------------------------
def test_model_path_store_norm_handles_none_and_invalid(tmp_path: Path):
    s = InMemoryQSettings()
    mps = store.ModelPathStore(settings=s)

    # None should normalize to None
    assert mps._norm_existing_path(None) is None  # type: ignore[arg-type]
    assert mps._norm_existing_dir(None) is None  # type: ignore[arg-type]

    # Existing dir should normalize to an absolute path
    d = tmp_path / "models"
    d.mkdir()
    norm_dir = mps._norm_existing_dir(str(d))
    assert norm_dir is not None
    assert Path(norm_dir).exists()
    assert Path(norm_dir).is_dir()

    # Existing file should normalize as existing path
    f = d / "net.pt"
    f.write_text("x")
    norm_file = mps._norm_existing_path(str(f))
    assert norm_file is not None
    assert Path(norm_file).exists()
    assert Path(norm_file).is_file()


# -----------------------------
# ModelPathStore: load/save
# -----------------------------
def test_model_path_store_load_last_valid_model_file(tmp_path: Path):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    model = tmp_path / "model.pt"
    model.write_text("x")

    settings.setValue("dlc/last_model_path", str(model))

    assert mps.load_last() == str(model)


def test_model_path_store_load_last_invalid_extension_returns_none(tmp_path: Path):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    bad = tmp_path / "model.onnx"
    bad.write_text("x")

    settings.setValue("dlc/last_model_path", str(bad))
    assert mps.load_last() is None


def test_model_path_store_load_last_missing_file_returns_none(tmp_path: Path):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    missing = tmp_path / "missing.pt"
    settings.setValue("dlc/last_model_path", str(missing))
    assert mps.load_last() is None


def test_model_path_store_load_last_dir_valid(tmp_path: Path):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    d = tmp_path / "models"
    d.mkdir()

    settings.setValue("dlc/last_model_dir", str(d))
    assert mps.load_last_dir() == str(d)


def test_model_path_store_load_last_dir_invalid(tmp_path: Path):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    missing = tmp_path / "nope"
    settings.setValue("dlc/last_model_dir", str(missing))
    assert mps.load_last_dir() is None


def test_model_path_store_save_if_valid_saves_dir_always_and_file_only_if_valid(tmp_path: Path):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    d = tmp_path / "models"
    d.mkdir()

    valid = d / "net.pth"
    valid.write_text("x")

    invalid = d / "net.onnx"
    invalid.write_text("x")

    # Save invalid first: should save last_model_dir but not last_model_path
    mps.save_if_valid(str(invalid))
    assert settings.value("dlc/last_model_dir") == str(d)
    assert settings.value("dlc/last_model_path", "") in ("", None)

    # Save valid: should save both
    mps.save_if_valid(str(valid))
    assert settings.value("dlc/last_model_dir") == str(d)
    assert settings.value("dlc/last_model_path") == str(valid)


def test_model_path_store_save_last_dir_only_saves_when_dir_exists(tmp_path: Path):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    good = tmp_path / "good"
    good.mkdir()
    bad = tmp_path / "bad"

    mps.save_last_dir(str(bad))
    assert settings.value("dlc/last_model_dir", "") in ("", None)

    mps.save_last_dir(str(good))
    assert settings.value("dlc/last_model_dir") == str(good)


# -----------------------------
# ModelPathStore: resolve
# -----------------------------
def test_model_path_store_resolve_prefers_config_path_when_valid(tmp_path: Path):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    model = tmp_path / "cfg_model.pt"
    model.write_text("x")

    # Persisted points to something else; config_path should win
    other = tmp_path / "other.pt"
    other.write_text("x")
    settings.setValue("dlc/last_model_path", str(other))

    assert mps.resolve(str(model)) == str(model)


def test_model_path_store_resolve_falls_back_to_persisted(tmp_path: Path):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    persisted = tmp_path / "persisted.pb"
    persisted.write_text("x")
    settings.setValue("dlc/last_model_path", str(persisted))

    # invalid config path
    bad = tmp_path / "notamodel.onnx"
    bad.write_text("x")

    assert mps.resolve(str(bad)) == str(persisted)


def test_model_path_store_resolve_returns_empty_when_nothing_valid(tmp_path: Path):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    assert mps.resolve(None) == ""
    assert mps.resolve("") == ""


# -----------------------------
# ModelPathStore: suggest_start_dir
# -----------------------------
def test_model_path_store_suggest_start_dir_prefers_last_dir(tmp_path: Path):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    d = tmp_path / "lastdir"
    d.mkdir()
    settings.setValue("dlc/last_model_dir", str(d))

    assert mps.suggest_start_dir(fallback_dir=str(tmp_path)) == str(d)


def test_model_path_store_suggest_start_dir_uses_parent_of_last_file(tmp_path: Path):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    d = tmp_path / "models"
    d.mkdir()
    model = d / "net.pt"
    model.write_text("x")

    settings.setValue("dlc/last_model_path", str(model))

    assert mps.suggest_start_dir(fallback_dir=str(tmp_path / "fallback")) == str(d)


def test_model_path_store_suggest_start_dir_uses_fallback_dir_if_valid(tmp_path: Path):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    fallback = tmp_path / "fallback"
    fallback.mkdir()

    assert mps.suggest_start_dir(fallback_dir=str(fallback)) == str(fallback)


def test_model_path_store_suggest_start_dir_falls_back_to_home(tmp_path: Path, monkeypatch):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    fake_home = tmp_path / "home"
    fake_home.mkdir()

    # Make cwd "invalid" so suggest_start_dir can't use it
    fake_cwd = tmp_path / "does_not_exist"
    assert not fake_cwd.exists()

    monkeypatch.setattr(store.Path, "home", lambda: fake_home)
    monkeypatch.setattr(store.Path, "cwd", lambda: fake_cwd)

    assert mps.suggest_start_dir(fallback_dir=None) == str(fake_home)


# -----------------------------
# ModelPathStore: suggest_selected_file
# -----------------------------
def test_model_path_store_suggest_selected_file_returns_existing_file(tmp_path: Path):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    model = tmp_path / "net.pt"
    model.write_text("x")
    settings.setValue("dlc/last_model_path", str(model))

    assert mps.suggest_selected_file() == str(model)


def test_model_path_store_suggest_selected_file_returns_none_when_missing(tmp_path: Path):
    settings = InMemoryQSettings()
    mps = store.ModelPathStore(settings=settings)

    missing = tmp_path / "missing.pt"
    settings.setValue("dlc/last_model_path", str(missing))

    assert mps.suggest_selected_file() is None
