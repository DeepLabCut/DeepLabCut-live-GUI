from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

from PySide6.QtCore import QObject, Signal


def poet_default_weights_dir() -> Path:
    return Path(tempfile.gettempdir()) / "dlclivegui" / "poet"


POET_WEIGHTS_URL = "https://zenodo.org/records/7972042/files/poet_ckpt.pth?download=1"
POET_WEIGHTS_FILENAME = "poet_resnet50.pth"


class WeightsDownloadWorker(QObject):
    progress = Signal(int)  # 0..100
    finished = Signal(str)  # path
    error = Signal(str)

    def __init__(self, url: str, dest: Path):
        super().__init__()
        self.url = url
        self.dest = dest

    def run(self) -> None:
        try:
            self.dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.dest.with_suffix(self.dest.suffix + ".part")

            req = urllib.request.Request(self.url, headers={"User-Agent": "DLCLiveGUI"})
            with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
                total = resp.length or 0
                done = 0
                chunk = 1024 * 256

                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if total > 0:
                        self.progress.emit(int(done * 100 / total))

            final_path = self.dest.parent / POET_WEIGHTS_FILENAME
            os.replace(tmp, final_path)
            self.progress.emit(100)
            self.finished.emit(str(final_path))

        except Exception as e:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            self.error.emit(str(e))
