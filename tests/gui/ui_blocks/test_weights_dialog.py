from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog

from dlclivegui.gui.misc.weights_dialog import PoetWeightsDialog


def test_dialog_without_initial_path_disables_use_button(
    qtbot,
) -> None:
    dialog = PoetWeightsDialog()
    qtbot.addWidget(dialog)

    assert dialog.path_label.text() == "No weights selected"
    assert not dialog.use_button.isEnabled()
    assert not dialog.progress_bar.isVisible()


def test_dialog_accepts_valid_initial_path(
    qtbot,
    tmp_path,
) -> None:
    path = tmp_path / "weights.pth"
    path.touch()

    dialog = PoetWeightsDialog(initial_path=str(path))
    qtbot.addWidget(dialog)

    assert dialog.path_label.text() == str(path)
    assert dialog.use_button.isEnabled()


def test_use_selected_emits_path(
    qtbot,
    tmp_path,
) -> None:
    path = tmp_path / "weights.pth"
    path.touch()

    dialog = PoetWeightsDialog(initial_path=str(path))
    qtbot.addWidget(dialog)

    with qtbot.waitSignal(
        dialog.weights_selected,
        timeout=1000,
    ) as blocker:
        qtbot.mouseClick(
            dialog.use_button,
            Qt.MouseButton.LeftButton,
        )

    assert blocker.args == [str(path)]


def test_browse_sets_selected_path(
    qtbot,
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "weights.pth"
    path.touch()

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (
            str(path),
            "POET weights (*.pt *.pth)",
        ),
    )

    dialog = PoetWeightsDialog()
    qtbot.addWidget(dialog)

    dialog._browse()

    assert dialog.path_label.text() == str(path)
    assert dialog.use_button.isEnabled()


def test_download_cleanup_restores_controls(
    qtbot,
    tmp_path,
) -> None:
    path = tmp_path / "weights.pth"
    path.touch()

    dialog = PoetWeightsDialog()
    qtbot.addWidget(dialog)

    dialog._set_selected_path(str(path))
    dialog.browse_button.setEnabled(False)
    dialog.download_button.setEnabled(False)

    dialog._download_cleanup()

    assert dialog.browse_button.isEnabled()
    assert dialog.download_button.isEnabled()
    assert dialog.use_button.isEnabled()
