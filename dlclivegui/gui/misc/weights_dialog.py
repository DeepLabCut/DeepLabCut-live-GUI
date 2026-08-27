"""Dialog for selecting or downloading POET weights."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from dlclivegui.services.inference.models.poet.weights import (
    POET_WEIGHTS_FILENAME,
    POET_WEIGHTS_URL,
    WeightsDownloadWorker,
    poet_default_weights_dir,
)


class PoetWeightsDialog(QDialog):
    """Select existing POET weights or download the default checkpoint."""

    weights_selected = Signal(str)

    def __init__(
        self,
        parent=None,
        *,
        initial_path: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("POET weights")
        self.setModal(False)

        self._selected_path = initial_path
        self._download_thread: QThread | None = None
        self._download_worker: WeightsDownloadWorker | None = None

        self.path_label = QLabel(initial_path or "No weights selected")
        self.path_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        self.browse_button = QPushButton("Select existing weights...")
        self.download_button = QPushButton("Download default weights")
        self.use_button = QPushButton("Use selected weights")
        self.use_button.setEnabled(self._is_valid_weights_path(initial_path))

        button_row = QHBoxLayout()
        button_row.addWidget(self.browse_button)
        button_row.addWidget(self.download_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.path_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(button_row)
        layout.addWidget(self.use_button)

        self.browse_button.clicked.connect(self._browse)
        self.download_button.clicked.connect(self._download)
        self.use_button.clicked.connect(self._use_selected)

    @staticmethod
    def _is_valid_weights_path(
        path: str,
    ) -> bool:
        candidate = Path(path).expanduser()

        return candidate.is_file() and candidate.suffix.lower() in {
            ".pt",
            ".pth",
        }

    @Slot()
    def _browse(self) -> None:
        start_path = (
            self._selected_path if self._is_valid_weights_path(self._selected_path) else str(poet_default_weights_dir())
        )

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select POET weights",
            start_path,
            "POET weights (*.pt *.pth)",
        )

        if path:
            self._set_selected_path(path)

    @Slot()
    def _download(self) -> None:
        if self._download_thread is not None:
            return

        destination = poet_default_weights_dir() / POET_WEIGHTS_FILENAME

        thread = QThread(self)
        worker = WeightsDownloadWorker(
            POET_WEIGHTS_URL,
            destination,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self.progress_bar.setValue)
        worker.finished.connect(self._download_finished)
        worker.error.connect(self._download_failed)

        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)

        thread.finished.connect(self._download_cleanup)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._download_thread = thread
        self._download_worker = worker

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.browse_button.setEnabled(False)
        self.download_button.setEnabled(False)
        self.use_button.setEnabled(False)

        thread.start()

    @Slot(str)
    def _download_finished(
        self,
        path: str,
    ) -> None:
        self.progress_bar.setValue(100)
        self._set_selected_path(path)

    @Slot(str)
    def _download_failed(
        self,
        message: str,
    ) -> None:
        QMessageBox.critical(
            self,
            "POET weights download failed",
            message,
        )

    @Slot()
    def _download_cleanup(self) -> None:
        self._download_thread = None
        self._download_worker = None

        self.browse_button.setEnabled(True)
        self.download_button.setEnabled(True)
        self.use_button.setEnabled(self._is_valid_weights_path(self._selected_path))

    def _set_selected_path(
        self,
        path: str,
    ) -> None:
        self._selected_path = path
        self.path_label.setText(path)
        self.use_button.setEnabled(self._is_valid_weights_path(path))

    @Slot()
    def _use_selected(self) -> None:
        if not self._is_valid_weights_path(self._selected_path):
            return

        self.weights_selected.emit(self._selected_path)
        self.accept()
