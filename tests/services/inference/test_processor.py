from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest

from dlclivegui.services.dlc_processor import DLCLiveProcessor
from dlclivegui.services.inference.base import (
    PoseBackend,
    PoseBackends,
    PosePacket,
    PoseSource,
)
from dlclivegui.services.inference.processor import PoseProcessor, create_pose_processor


class FakePoseBackend(PoseBackend):
    def __init__(
        self,
        *,
        pose: np.ndarray | None = None,
        initialization_error: Exception | None = None,
        inference_error: Exception | None = None,
    ) -> None:
        self.pose = pose
        self.initialization_error = initialization_error
        self.inference_error = inference_error

        self.initialized_with: np.ndarray | None = None
        self.received_frames: list[np.ndarray] = []
        self.closed = False

    def init_inference(
        self,
        init_frame: np.ndarray,
    ) -> None:
        if self.initialization_error is not None:
            raise self.initialization_error

        self.initialized_with = init_frame.copy()

    def get_pose(
        self,
        frame: np.ndarray,
        frame_time: float | None = None,
    ) -> np.ndarray | None:
        del frame_time

        if self.inference_error is not None:
            raise self.inference_error

        self.received_frames.append(frame.copy())
        return self.pose

    def make_pose_packet(
        self,
        pose: np.ndarray | None,
    ) -> PosePacket:
        return PosePacket(
            schema_version=1,
            keypoints=pose,
            keypoint_names=["a", "b"],
            individual_ids=None,
            skeleton_id="test.line",
            skeleton_edges=(("a", "b"),),
            source=PoseSource(
                backend=PoseBackends.POET,
                model_type=None,
            ),
            raw=pose,
        )

    def close(self) -> None:
        self.closed = True


def test_configure_stores_backend_factory() -> None:
    processor = PoseProcessor()

    processor.configure(FakePoseBackend)

    assert processor.is_configured()


def test_process_frame_emits_pose_result_with_packet(
    qtbot,
) -> None:
    pose = np.array(
        [
            [1.0, 2.0, 0.9],
            [3.0, 4.0, 0.8],
        ],
        dtype=np.float32,
    )
    backend = FakePoseBackend(pose=pose)

    processor = PoseProcessor()
    processor._backend = backend

    with qtbot.waitSignal(
        processor.pose_ready,
        timeout=1000,
    ) as blocker:
        processor._process_frame(
            np.zeros((10, 10, 3), dtype=np.uint8),
            timestamp=10.0,
            enqueue_time=time.perf_counter(),
            queue_wait_time=0.0,
        )

    result = blocker.args[0]

    assert result.timestamp == 10.0
    assert result.pose is pose
    assert result.packet.skeleton_id == "test.line"
    assert result.packet.skeleton_edges == (("a", "b"),)


def test_enqueue_first_frame_starts_worker(
    qtbot,
) -> None:
    backend = FakePoseBackend(pose=np.zeros((2, 3), dtype=np.float32))
    processor = PoseProcessor()
    processor.configure(lambda: backend)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    with qtbot.waitSignal(
        processor.initialized,
        timeout=2000,
    ) as blocker:
        processor.enqueue_frame(frame, timestamp=1.0)

    assert blocker.args == [True]
    assert backend.initialized_with is not None

    processor.shutdown()
    assert backend.closed


def test_initialization_failure_emits_error_and_false(
    qtbot,
) -> None:
    processor = PoseProcessor()
    processor.configure(lambda: FakePoseBackend(initialization_error=RuntimeError("broken backend")))

    errors: list[str] = []
    initialized: list[bool] = []

    processor.error.connect(errors.append)
    processor.initialized.connect(initialized.append)

    processor.enqueue_frame(
        np.zeros((10, 10, 3), dtype=np.uint8),
        timestamp=1.0,
    )

    qtbot.waitUntil(
        lambda: bool(initialized),
        timeout=2000,
    )

    assert errors == ["broken backend"]
    assert initialized == [False]
    assert processor._backend is None
    assert processor._worker_thread is None
    assert processor._queue is None


def test_enqueue_frame_counts_drop_when_queue_is_full() -> None:
    processor = PoseProcessor()
    processor._worker_thread = threading.current_thread()

    processor._queue = queue.Queue(maxsize=1)
    processor._queue.put_nowait(
        (
            np.zeros((1, 1, 3), dtype=np.uint8),
            1.0,
            1.0,
        )
    )

    processor.enqueue_frame(
        np.zeros((1, 1, 3), dtype=np.uint8),
        timestamp=2.0,
    )

    stats = processor.get_stats()

    assert stats.frames_dropped == 1


def test_reset_clears_statistics() -> None:
    processor = PoseProcessor()
    processor._frames_enqueued = 4
    processor._frames_processed = 3
    processor._frames_dropped = 2
    processor._latencies.append(1.0)
    processor._processing_times.extend([1.0, 2.0])

    processor.reset()

    stats = processor.get_stats()

    assert stats.frames_enqueued == 0
    assert stats.frames_processed == 0
    assert stats.frames_dropped == 0
    assert stats.average_latency == 0.0
    assert stats.processing_fps == 0.0


def test_reset_returns_false_when_worker_does_not_stop(
    monkeypatch,
) -> None:
    processor = PoseProcessor()

    monkeypatch.setattr(
        processor,
        "_stop_worker",
        lambda: False,
    )

    stopped = processor.reset()

    assert stopped is False


def test_failed_reset_preserves_runtime_statistics(
    monkeypatch,
) -> None:
    processor = PoseProcessor()
    processor._frames_enqueued = 4
    processor._frames_processed = 3
    processor._frames_dropped = 2
    processor._latencies.append(1.0)

    monkeypatch.setattr(
        processor,
        "_stop_worker",
        lambda: False,
    )

    stopped = processor.reset()

    assert stopped is False

    stats = processor.get_stats()

    assert stats.frames_enqueued == 4
    assert stats.frames_processed == 3
    assert stats.frames_dropped == 2
    assert stats.average_latency == 1.0


def test_create_dlc_pose_processor() -> None:
    processor = create_pose_processor("dlc")

    assert isinstance(processor, DLCLiveProcessor)


def test_create_poet_pose_processor() -> None:
    processor = create_pose_processor("poet")

    assert isinstance(processor, PoseProcessor)


def test_create_pose_processor_rejects_unknown_backend() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported pose backend",
    ):
        create_pose_processor("unknown")
