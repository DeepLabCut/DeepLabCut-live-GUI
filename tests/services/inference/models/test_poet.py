from __future__ import annotations

import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
import pytest

from dlclivegui.services.inference.base import PoseBackends
from dlclivegui.services.inference.models.poet.poet_processor import (
    POET_KEYPOINT_NAMES,
    POET_SKELETON_EDGES,
    POET_SKELETON_ID,
    POETBackend,
)
from dlclivegui.services.inference.models.poet.weights import (
    WeightsDownloadWorker,
)

TEST_WEIGHTS_URL = "https://example.invalid/weights"


class FakeResponse(BytesIO):
    """In-memory response implementing the context-manager protocol."""

    def __init__(self, content: bytes) -> None:
        super().__init__(content)
        self.length = len(content)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()


@dataclass
class UrlopenStub:
    """Configurable replacement for urllib.request.urlopen."""

    content: bytes = b""
    error: Exception | None = None
    before_error: Callable[[], None] | None = None
    expected_timeout: float = 30
    calls: list[tuple[Any, float | None]] = field(default_factory=list)

    def __call__(
        self,
        request,
        *,
        timeout: float | None = None,
    ) -> FakeResponse:
        self.calls.append((request, timeout))

        assert timeout == self.expected_timeout

        if self.before_error is not None:
            self.before_error()

        if self.error is not None:
            raise self.error

        return FakeResponse(self.content)


@pytest.fixture
def urlopen_factory(
    monkeypatch,
) -> Callable[..., UrlopenStub]:
    """Install and return a configurable urlopen test stub."""

    def install(
        *,
        content: bytes = b"",
        error: Exception | None = None,
        before_error: Callable[[], None] | None = None,
        expected_timeout: float = 30,
    ) -> UrlopenStub:
        if error is None and before_error is not None:
            raise ValueError("before_error requires an error.")

        stub = UrlopenStub(
            content=content,
            error=error,
            before_error=before_error,
            expected_timeout=expected_timeout,
        )

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            stub,
        )

        return stub

    return install


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
    pose = np.zeros(
        (2, 17, 3),
        dtype=np.float32,
    )

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
    assert packet.skeleton_id == POET_SKELETON_ID
    assert packet.skeleton_edges == POET_SKELETON_EDGES


def test_download_worker_reuses_existing_file(
    qtbot,
    tmp_path,
) -> None:
    destination = tmp_path / "weights.pth"
    destination.write_bytes(b"existing")

    worker = WeightsDownloadWorker(
        TEST_WEIGHTS_URL,
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


def test_download_worker_writes_destination(
    qtbot,
    tmp_path,
    urlopen_factory,
) -> None:
    destination = tmp_path / "weights.pth"
    content = b"model-data"

    urlopen_stub = urlopen_factory(
        content=content,
    )

    worker = WeightsDownloadWorker(
        TEST_WEIGHTS_URL,
        destination,
    )

    errors: list[str] = []
    worker.error.connect(errors.append)

    with qtbot.waitSignal(
        worker.finished,
        timeout=1000,
    ) as blocker:
        worker.run()

    assert errors == []
    assert blocker.args == [str(destination)]
    assert destination.read_bytes() == content
    assert not destination.with_suffix(".pth.part").exists()

    assert len(urlopen_stub.calls) == 1
    request, timeout = urlopen_stub.calls[0]

    assert isinstance(
        request,
        urllib.request.Request,
    )
    assert request.full_url == TEST_WEIGHTS_URL
    assert timeout == 30


def test_download_worker_reports_failure_and_cleans_partial_file(
    qtbot,
    tmp_path,
    urlopen_factory,
) -> None:
    destination = tmp_path / "weights.pth"
    partial_path = destination.with_suffix(".pth.part")

    urlopen_stub = urlopen_factory(
        error=OSError("download failed"),
        before_error=lambda: partial_path.write_bytes(b"partial"),
    )

    worker = WeightsDownloadWorker(
        TEST_WEIGHTS_URL,
        destination,
    )

    finished_paths: list[str] = []
    worker.finished.connect(finished_paths.append)

    with qtbot.waitSignal(
        worker.error,
        timeout=1000,
    ) as blocker:
        worker.run()

    assert blocker.args == ["download failed"]
    assert finished_paths == []
    assert not partial_path.exists()
    assert not destination.exists()

    assert len(urlopen_stub.calls) == 1
    request, timeout = urlopen_stub.calls[0]

    assert isinstance(
        request,
        urllib.request.Request,
    )
    assert request.full_url == TEST_WEIGHTS_URL
    assert timeout == 30
