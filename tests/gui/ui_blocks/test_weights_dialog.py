from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from dlclivegui.gui.misc.weights_dialog import PoetWeightsDialog


def test_dialog_shows_download_destination(
    qtbot,
) -> None:
    dialog = PoetWeightsDialog()
    qtbot.addWidget(dialog)

    assert dialog.path_label.full_text == str(dialog._destination)
    assert dialog.path_label.text()
    assert dialog.path_label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    assert dialog.status_label.text() == "Ready to download."
    assert dialog.progress_bar.value() == 0
    assert dialog.download_button.isEnabled()
    assert dialog.close_button.isEnabled()


def test_update_progress_updates_progress_and_status(
    qtbot,
) -> None:
    dialog = PoetWeightsDialog()
    qtbot.addWidget(dialog)

    dialog._update_progress(42)

    assert dialog.progress_bar.value() == 42
    assert dialog.status_label.text() == "Downloading POET weights... 42%"


def test_download_finished_records_completed_path(
    qtbot,
    tmp_path,
) -> None:
    path = tmp_path / "weights.pth"
    path.touch()

    dialog = PoetWeightsDialog()
    qtbot.addWidget(dialog)

    dialog._download_finished(str(path))

    assert dialog._completed_path == str(path)
    assert dialog.progress_bar.value() == 100
    assert dialog.status_label.text() == ("Download complete. Returning to the main window...")


def test_download_cleanup_emits_path_and_accepts(
    qtbot,
    tmp_path,
) -> None:
    path = tmp_path / "weights.pth"
    path.touch()

    dialog = PoetWeightsDialog()
    qtbot.addWidget(dialog)
    dialog._completed_path = str(path)

    with qtbot.waitSignal(
        dialog.weights_downloaded,
        timeout=1000,
    ) as blocker:
        dialog._download_cleanup()

    assert blocker.args == [str(path)]
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog._download_thread is None
    assert dialog._download_worker is None


def test_download_cleanup_after_failure_restores_controls(
    qtbot,
) -> None:
    dialog = PoetWeightsDialog()
    qtbot.addWidget(dialog)

    dialog.download_button.setEnabled(False)
    dialog.close_button.setEnabled(False)
    dialog._completed_path = None

    dialog._download_cleanup()

    assert dialog.download_button.isEnabled()
    assert dialog.close_button.isEnabled()


def test_reject_is_ignored_while_download_is_active(
    qtbot,
) -> None:
    dialog = PoetWeightsDialog()
    qtbot.addWidget(dialog)

    dialog._download_thread = object()

    dialog.reject()

    assert dialog.result() == 0


def test_reject_closes_when_idle(
    qtbot,
) -> None:
    dialog = PoetWeightsDialog()
    qtbot.addWidget(dialog)

    dialog.reject()

    assert dialog.result() == QDialog.DialogCode.Rejected
