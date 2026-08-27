"""Dialog for downloading the default POET weights."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from dlclivegui.gui.misc.eliding_label import ElidingPathLabel
from dlclivegui.services.inference.models.poet.weights import (
    POET_WEIGHTS_FILENAME,
    POET_WEIGHTS_URL,
    WeightsDownloadWorker,
    poet_default_weights_dir,
)


class PoetWeightsDialog(QDialog):
    """Download the default POET checkpoint and return its path."""

    weights_downloaded = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Download POET weights")
        self.setModal(False)
        self.setMinimumWidth(480)

        self._destination = poet_default_weights_dir() / POET_WEIGHTS_FILENAME
        self._download_thread: QThread | None = None
        self._download_worker: WeightsDownloadWorker | None = None
        self._completed_path: str | None = None

        description = QLabel(
            "Download the default POET checkpoint. When the download "
            "finishes, the model will be selected automatically and "
            "this window will close."
        )
        description.setWordWrap(True)

        destination_title = QLabel("Destination:")

        self.path_label = ElidingPathLabel(str(self._destination))
        self.path_label.setToolTip(f"Click to copy:\n{self._destination}")

        self.status_label = QLabel("Ready to download.")
        self.status_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        self.download_button = QPushButton("Download weights")
        self.close_button = QPushButton("Close")

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.close_button)
        button_row.addWidget(self.download_button)

        layout = QVBoxLayout(self)
        layout.addWidget(description)
        layout.addSpacing(6)
        layout.addWidget(destination_title)
        layout.addWidget(self.path_label)
        layout.addSpacing(6)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(button_row)

        self.download_button.clicked.connect(self._download)
        self.close_button.clicked.connect(self.reject)

    @Slot()
    def _download(self) -> None:
        if self._download_thread is not None:
            return

        thread = QThread(self)
        worker = WeightsDownloadWorker(
            POET_WEIGHTS_URL,
            self._destination,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)

        worker.progress.connect(self._update_progress)
        worker.finished.connect(self._download_finished)
        worker.error.connect(self._download_failed)

        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)

        thread.finished.connect(self._download_cleanup)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._download_thread = thread
        self._download_worker = worker
        self._completed_path = None

        self.status_label.setText("Downloading POET weights...")
        self.progress_bar.setValue(0)

        self.download_button.setEnabled(False)
        self.close_button.setEnabled(False)

        thread.start()

    @Slot(int)
    def _update_progress(
        self,
        value: int,
    ) -> None:
        self.progress_bar.setValue(value)
        self.status_label.setText(f"Downloading POET weights... {value}%")

    @Slot(str)
    def _download_finished(
        self,
        path: str,
    ) -> None:
        self._completed_path = path
        self.progress_bar.setValue(100)
        self.status_label.setText("Download complete. Returning to the main window...")

    @Slot(str)
    def _download_failed(
        self,
        message: str,
    ) -> None:
        self._completed_path = None
        self.status_label.setText("Download failed.")

        QMessageBox.critical(
            self,
            "POET weights download failed",
            message,
        )

    @Slot()
    def _download_cleanup(self) -> None:
        self._download_thread = None
        self._download_worker = None

        completed_path = self._completed_path
        self._completed_path = None

        if completed_path is not None:
            self.weights_downloaded.emit(completed_path)
            self.accept()
            return

        self.download_button.setEnabled(True)
        self.close_button.setEnabled(True)

    def reject(self) -> None:
        if self._download_thread is not None:
            return

        super().reject()
