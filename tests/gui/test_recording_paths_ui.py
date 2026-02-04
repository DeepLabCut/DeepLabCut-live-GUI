# tests/gui/test_recording_paths_ui.py
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt

from dlclivegui.gui.main_window import DLCLiveMainWindow


def test_recording_path_preview_updates(window, qtbot, tmp_path):
    # baseline: should be set after apply_config()
    assert window.recording_path_preview.text() != ""

    # Set output dir
    out_dir = tmp_path / "out"
    window.output_directory_edit.setText(str(out_dir))

    # Set session name + filename + container
    window.session_name_edit.setText("mouseA_day1")
    window.filename_edit.setText("trial01")
    window.container_combo.setCurrentText("avi")

    # Timestamp ON -> should show run_<timestamp>
    window.use_timestamp_checkbox.setChecked(True)
    qtbot.wait(10)  # allow queued signals

    txt = window.recording_path_preview.text()
    assert "mouseA_day1" in txt
    assert "run_" in txt
    assert "timestamp" in txt  # label contains run_&lt;timestamp&gt; in your code
    assert "trial01" in txt
    assert ".avi" in txt

    # Timestamp OFF -> should show run_<next>
    window.use_timestamp_checkbox.setChecked(False)
    qtbot.wait(10)

    txt2 = window.recording_path_preview.text()
    assert "mouseA_day1" in txt2
    assert "run_" in txt2
    assert "next" in txt2
    assert ".avi" in txt2


def test_session_name_persists_across_windows(qtbot, app_config_two_cams):
    # First window: set session name and persist via editingFinished handler
    w1 = DLCLiveMainWindow(config=app_config_two_cams)
    qtbot.addWidget(w1)
    w1.setAttribute(Qt.WA_DontShowOnScreen, True)
    w1.show()

    w1.session_name_edit.setText("persist_me")
    w1._on_session_name_editing_finished()  # deterministic persistence
    w1.close()

    # Second window: should restore persisted session name
    w2 = DLCLiveMainWindow(config=app_config_two_cams)
    qtbot.addWidget(w2)
    w2.setAttribute(Qt.WA_DontShowOnScreen, True)
    w2.show()

    assert w2.session_name_edit.text() == "persist_me"
    w2.close()


def test_use_timestamp_persists_across_windows(qtbot, app_config_two_cams):
    w1 = DLCLiveMainWindow(config=app_config_two_cams)
    qtbot.addWidget(w1)
    w1.setAttribute(Qt.WA_DontShowOnScreen, True)
    w1.show()

    # toggle off and persist using your handler
    w1.use_timestamp_checkbox.setChecked(False)
    w1._on_use_timestamp_changed(0)
    w1.close()

    # new window restores False
    w2 = DLCLiveMainWindow(config=app_config_two_cams)
    qtbot.addWidget(w2)
    w2.setAttribute(Qt.WA_DontShowOnScreen, True)
    w2.show()

    assert w2.use_timestamp_checkbox.isChecked() is False
    w2.close()


def test_start_recording_passes_session_and_timestamp(window, start_all_spy, qtbot):
    window.session_name_edit.setText("Sess42")
    window.use_timestamp_checkbox.setChecked(False)

    # No need to start preview; your _start_multi_camera_recording only requires active_cams
    window._start_multi_camera_recording()

    kwargs = start_all_spy["kwargs"]
    assert kwargs["session_name"] == "Sess42"
    assert kwargs["use_timestamp"] is False
    assert "all_or_nothing" in kwargs
    # Ensure recording.directory and recording.filename match UI
    recording = start_all_spy["recording"]
    assert recording.output_path().parent == Path(window.output_directory_edit.text()).expanduser().resolve()
    assert recording.container == window.container_combo.currentText()
    assert recording.codec == window.codec_combo.currentText()
    assert recording.filename == window.filename_edit.text()


def test_processor_overrides_session_name_and_persists(window, start_all_spy, monkeypatch, fake_processor):
    # Arrange window state so processor status logic runs
    window._dlc_active = True
    window._dlc_initialized = True
    window.auto_record_checkbox.setChecked(True)

    # Patch start_recording to avoid preview start/timers
    monkeypatch.setattr(window, "_start_recording", lambda: window._start_multi_camera_recording())

    # Install fake processor
    window._dlc._processor = fake_processor
    window._last_processor_vid_recording = False  # ensure it sees a "change"

    # Act
    window._update_processor_status()

    # Assert UI updated
    assert window.session_name_edit.text() == "auto_ABC"
    assert window.filename_edit.text() == "auto_ABC"

    # Assert recording call used overridden session name
    kwargs = start_all_spy["kwargs"]
    assert kwargs["session_name"] == "auto_ABC"
