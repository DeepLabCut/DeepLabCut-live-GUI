import urllib.request
from io import BytesIO

import numpy as np
import pytest

from dlclivegui.services.inference.base import PoseBackends
from dlclivegui.services.inference.models.poet.poet_processor import (
    POET_KEYPOINT_NAMES,
    POET_SKELETON_EDGES,
    POET_SKELETON_ID,
    POETBackend,
)
from dlclivegui.services.inference.models.poet.weights import WeightsDownloadWorker


def test_poet_backend_rejects_missing_checkpoint(
    tmp_path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="checkpoint not found",
    ):
        POETBackend(str(tmp_path / "missing.pth"))


def test_poet_backend_rejects_unsupported_extension(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "weights.txt"
    checkpoint.touch()

    with pytest.raises(
        ValueError,
        match=r"\.pt or \.pth",
    ):
        POETBackend(str(checkpoint))


def test_make_pose_packet_contains_poet_metadata(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "weights.pth"
    checkpoint.touch()

    backend = POETBackend(str(checkpoint))
    pose = np.zeros((2, 17, 3), dtype=np.float32)

    packet = backend.make_pose_packet(pose)

    assert packet.keypoints is pose
    assert packet.keypoint_names == list(POET_KEYPOINT_NAMES)
    assert packet.skeleton_id == POET_SKELETON_ID
    assert packet.skeleton_edges == POET_SKELETON_EDGES
    assert packet.individual_ids == [
        "person_0",
        "person_1",
    ]
    assert packet.source.backend == PoseBackends.POET


def test_make_pose_packet_without_detections_keeps_metadata(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "weights.pth"
    checkpoint.touch()

    backend = POETBackend(str(checkpoint))

    packet = backend.make_pose_packet(None)

    assert packet.keypoints is None
    assert packet.individual_ids == []
    assert packet.keypoint_names == list(POET_KEYPOINT_NAMES)
    assert packet.skeleton_edges == POET_SKELETON_EDGES


def test_download_worker_reuses_existing_file(
    qtbot,
    tmp_path,
) -> None:
    destination = tmp_path / "weights.pth"
    destination.write_bytes(b"existing")

    worker = WeightsDownloadWorker(
        "https://example.invalid/weights",
        destination,
    )

    progress: list[int] = []

    worker.progress.connect(progress.append)

    with qtbot.waitSignal(
        worker.finished,
        timeout=1000,
    ) as blocker:
        worker.run()

    assert blocker.args == [str(destination)]
    assert progress == [100]
    assert destination.read_bytes() == b"existing"


class FakeResponse(BytesIO):
    def __init__(self, content: bytes) -> None:
        super().__init__(content)
        self.length = len(content)

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()


def test_download_worker_writes_destination(
    qtbot,
    monkeypatch,
    tmp_path,
) -> None:
    destination = tmp_path / "weights.pth"
    content = b"model-data"

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda _request: FakeResponse(content),
    )

    worker = WeightsDownloadWorker(
        "https://example.invalid/weights",
        destination,
    )

    with qtbot.waitSignal(
        worker.finished,
        timeout=1000,
    ) as blocker:
        worker.run()

    assert blocker.args == [str(destination)]
    assert destination.read_bytes() == content
    assert not destination.with_suffix(".pth.part").exists()


def test_download_worker_reports_failure_and_cleans_partial_file(
    qtbot,
    monkeypatch,
    tmp_path,
) -> None:
    destination = tmp_path / "weights.pth"
    partial_path = destination.with_suffix(".pth.part")

    def failing_urlopen(_request):
        partial_path.write_bytes(b"partial")
        raise OSError("download failed")

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        failing_urlopen,
    )

    worker = WeightsDownloadWorker(
        "https://example.invalid/weights",
        destination,
    )

    with qtbot.waitSignal(
        worker.error,
        timeout=1000,
    ) as blocker:
        worker.run()

    assert blocker.args == ["download failed"]
    assert not partial_path.exists()
    assert not destination.exists()
