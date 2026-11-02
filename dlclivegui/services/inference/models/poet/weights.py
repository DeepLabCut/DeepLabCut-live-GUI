from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger(__name__)


POET_WEIGHTS_URL = "https://zenodo.org/records/7972042/files/poet_ckpt.pth?download=1"
POET_WEIGHTS_FILENAME = "poet_resnet50.pth"


def poet_default_weights_dir() -> Path:
    return Path.home() / ".cache/dlclivegui/poet"


class WeightsDownloadWorker(QObject):
    progress = Signal(int)  # 0..100
    finished = Signal(str)  # path
    error = Signal(str)

    def __init__(self, url: str, dest: Path):
        super().__init__()
        self.url = url
        self.dest = dest

    def run(self) -> None:
        tmp = None

        try:
            self.dest.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if self.dest.is_file():
                self.progress.emit(100)
                self.finished.emit(str(self.dest))
                return

            tmp = self.dest.with_suffix(self.dest.suffix + ".part")
            request = urllib.request.Request(
                self.url,
                headers={"User-Agent": "DLCLiveGUI"},
            )
            with (
                urllib.request.urlopen(
                    request,
                    timeout=30,
                ) as response,
                tmp.open("wb") as output,
            ):
                total = response.length or 0
                downloaded = 0
                chunk_size = 256 * 1024

                while True:
                    if QThread.currentThread().isInterruptionRequested():
                        raise RuntimeError("POET weights download was cancelled.")

                    chunk = response.read(chunk_size)
                    if not chunk:
                        break

                    output.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        self.progress.emit(int(downloaded * 100 / total))

            os.replace(tmp, self.dest)
            self.progress.emit(100)
            self.finished.emit(str(self.dest))

        except Exception as exc:
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Failed to remove partial POET weights.")

            self.error.emit(str(exc))
