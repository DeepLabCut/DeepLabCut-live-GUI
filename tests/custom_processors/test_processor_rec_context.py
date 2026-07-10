# tests/processors/test_processor_recording_context.py
from __future__ import annotations

import importlib
import pickle
import sys
import types
from types import SimpleNamespace

import pytest

# -----------------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------------


def _mock_dlclive(monkeypatch):
    """Install a tiny dlclive.processor.Processor mock before importing processors."""

    class Processor:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, pose, **kwargs):
            return pose

    dlclive_mod = types.ModuleType("dlclive")
    processor_mod = types.ModuleType("dlclive.processor")

    dlclive_mod.Processor = Processor
    processor_mod.Processor = Processor

    monkeypatch.setitem(sys.modules, "dlclive", dlclive_mod)
    monkeypatch.setitem(sys.modules, "dlclive.processor", processor_mod)


@pytest.fixture
def socket_mod(monkeypatch):
    """Import dlclivegui.processors.dlc_processor_socket with dlclive mocked."""
    _mock_dlclive(monkeypatch)
    mod_name = "dlclivegui.processors.dlc_processor_socket"
    sys.modules.pop(mod_name, None)
    return importlib.import_module(mod_name)


class DummyLineEdit:
    def __init__(self, value: str):
        self._value = value

    def text(self) -> str:
        return self._value


class HookProcessor:
    def __init__(self):
        self.started_contexts = []
        self.stopped_contexts = []
        self.save_calls = 0

    def on_recording_started(self, context):
        self.started_contexts.append(context)

    def on_recording_stopped(self, context):
        self.stopped_contexts.append(context)

    def save(self):
        self.save_calls += 1
        return 1


class SaveOnlyProcessor:
    def __init__(self):
        self.save_calls = 0

    def save(self):
        self.save_calls += 1
        return 1


# -----------------------------------------------------------------------------
# BaseProcessorSocket generic recording-context API
# -----------------------------------------------------------------------------


def test_base_processor_recording_context_sets_save_path(socket_mod, tmp_path):
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0))

    try:
        base_path = tmp_path / "MouseA_2026-07-10_1"
        context = {
            "run_dir": tmp_path,
            "session_name": "MouseA",
            "filename": "MouseA_2026-07-10_1.avi",
            "filename_stem": "MouseA_2026-07-10_1",
            "processor_base_path": base_path,
        }

        proc.set_recording_context(context)

        assert proc.recording_context == context
        assert proc.get_save_path() == base_path
        assert proc.save_path == base_path

    finally:
        proc.stop()


def test_base_processor_recording_hooks_update_context(socket_mod, tmp_path):
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0))

    try:
        started_context = {"processor_base_path": tmp_path / "started"}
        stopped_context = {"processor_base_path": tmp_path / "stopped"}

        proc.on_recording_started(started_context)
        assert proc.recording_context == started_context
        assert proc.get_save_path() == tmp_path / "started"

        proc.on_recording_stopped(stopped_context)
        assert proc.recording_context == stopped_context
        assert proc.get_save_path() == tmp_path / "stopped"

    finally:
        proc.stop()


def test_base_processor_save_uses_save_path_when_file_is_none(socket_mod, tmp_path):
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0), save_original=False)

    try:
        save_path = tmp_path / "legacy" / "MouseA_2026-07-10_1_PROC"
        proc.set_save_path(save_path)

        proc.start_recording()
        proc.process([[1.0, 2.0, 0.9]], frame_time=12.34, pose_time=12.35)
        proc.stop_recording()

        ret = proc.save()
        assert ret == 1
        assert save_path.exists()

        with save_path.open("rb") as f:
            payload = pickle.load(f)

        assert "time_stamp" in payload
        assert "step" in payload
        assert "frame_time" in payload
        assert len(payload["frame_time"]) == 1

    finally:
        proc.stop()


def test_base_processor_save_writes_to_explicit_absolute_file(socket_mod, tmp_path):
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0), save_original=False)

    try:
        explicit_path = tmp_path / "explicit" / "processor_data.pkl"
        ret = proc.save(explicit_path)

        assert ret == 1
        assert explicit_path.exists()

        with explicit_path.open("rb") as f:
            payload = pickle.load(f)

        assert "start_time" in payload

    finally:
        proc.stop()


def test_base_processor_save_without_file_or_save_path_returns_zero(socket_mod):
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0), save_original=False)

    try:
        proc.save_path = None
        assert proc.save() == 0
    finally:
        proc.stop()


def test_base_processor_stop_save_true_uses_save_path(socket_mod, tmp_path):
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0), save_original=False)

    save_path = tmp_path / "processor_on_stop.pkl"
    proc.set_save_path(save_path)

    # stop(save=True) should save, close listener, and be idempotent.
    proc.stop(save=True)
    assert save_path.exists()
    assert proc.listener is None

    proc.stop(save=True)  # no crash


# -----------------------------------------------------------------------------
# DLCLiveMainWindow recording-context helpers
# -----------------------------------------------------------------------------


@pytest.fixture
def main_window_cls():
    pytest.importorskip("PySide6")
    mod = importlib.import_module("dlclivegui.gui.main_window")
    return mod.DLCLiveMainWindow


def make_window_shell(main_window_cls, processor=None, run_dir=None):
    """Create a DLCLiveMainWindow shell without running QMainWindow.__init__."""
    win = main_window_cls.__new__(main_window_cls)
    win._dlc = SimpleNamespace(_processor=processor, _dlc=None)
    win._rec_manager = SimpleNamespace(run_dir=run_dir)
    win.session_name_edit = DummyLineEdit("MouseA")
    win.filename_edit = DummyLineEdit("MouseA_2026-07-10_1.avi")
    return win


def test_main_window_build_processor_recording_context(main_window_cls, tmp_path):
    win = make_window_shell(main_window_cls, run_dir=tmp_path)

    context = win._build_processor_recording_context(tmp_path)

    assert context["run_dir"] == tmp_path
    assert context["session_name"] == "MouseA"
    assert context["filename"] == "MouseA_2026-07-10_1.avi"
    assert context["filename_stem"] == "MouseA_2026-07-10_1"
    assert context["processor_base_path"] == tmp_path / "MouseA_2026-07-10_1"


def test_main_window_get_processor_instance_prefers_direct_processor(main_window_cls):
    processor = HookProcessor()
    win = make_window_shell(main_window_cls, processor=processor)

    assert win._get_dlc_processor_instance() is processor


def test_main_window_get_processor_instance_falls_back_to_dlclive_processor(main_window_cls):
    processor = HookProcessor()
    win = main_window_cls.__new__(main_window_cls)
    win._dlc = SimpleNamespace(_processor=None, _dlc=SimpleNamespace(processor=processor))

    assert win._get_dlc_processor_instance() is processor


def test_main_window_notify_processor_recording_started_calls_hook(main_window_cls, tmp_path):
    processor = HookProcessor()
    win = make_window_shell(main_window_cls, processor=processor, run_dir=tmp_path)

    win._notify_processor_recording_started(tmp_path)

    assert len(processor.started_contexts) == 1
    context = processor.started_contexts[0]
    assert context["processor_base_path"] == tmp_path / "MouseA_2026-07-10_1"


def test_main_window_notify_processor_recording_stopped_calls_hook(main_window_cls, tmp_path):
    processor = HookProcessor()
    win = make_window_shell(main_window_cls, processor=processor, run_dir=tmp_path)

    win._notify_processor_recording_stopped()

    assert len(processor.stopped_contexts) == 1
    context = processor.stopped_contexts[0]
    assert context["processor_base_path"] == tmp_path / "MouseA_2026-07-10_1"


def test_main_window_save_processor_data_calls_save(main_window_cls, tmp_path):
    processor = SaveOnlyProcessor()
    win = make_window_shell(main_window_cls, processor=processor, run_dir=tmp_path)

    win._save_processor_data_if_available()

    assert processor.save_calls == 1


def test_main_window_processor_hooks_are_optional(main_window_cls, tmp_path):
    class NoHooks:
        pass

    win = make_window_shell(main_window_cls, processor=NoHooks(), run_dir=tmp_path)

    # Optional hooks/save absence should not crash.
    win._notify_processor_recording_started(tmp_path)
    win._notify_processor_recording_stopped()
    win._save_processor_data_if_available()
