from __future__ import annotations

import queue
import threading

import numpy as np
import pytest

from dlclivegui.config import DLCProcessorSettings

# from dlclivegui.config import DLCProcessorSettings
from dlclivegui.processors.processor_utils import ProcessorSpec
from dlclivegui.services.dlc_processor import (
    DLCLiveProcessor,
)
from dlclivegui.services.inference.base import (
    ProcessorStats,
    WorkerState,
)


class _AliveThread:
    def is_alive(self) -> bool:
        return True


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_configure_accepts_pydantic(settings_model, monkeypatch_dlclive):
    proc = DLCLiveProcessor()
    proc.configure(settings_model)

    assert isinstance(proc._settings, DLCProcessorSettings)
    assert proc._settings.model_path == "dummy.pt"


@pytest.mark.unit
def test_worker_initializes_on_first_frame(qtbot, monkeypatch_dlclive, settings_model):
    proc = DLCLiveProcessor()
    proc.configure(settings_model)

    try:
        # First enqueued frame triggers worker start + initialization.
        with qtbot.waitSignal(proc.initialized, timeout=1500) as init_blocker:
            proc.enqueue_frame(np.zeros((100, 100, 3), dtype=np.uint8), timestamp=1.0)

        assert init_blocker.args == [True]
        assert proc._initialized
        assert getattr(proc._dlc, "init_called", False)

        # Optional: also ensure the init pose was delivered
        qtbot.waitUntil(lambda: proc.get_stats().frames_processed >= 1, timeout=1500)

    finally:
        proc.reset()  # Ensure thread cleanup


@pytest.mark.unit
def test_processor_spec_builds_in_worker_and_is_cleaned(
    qtbot,
    monkeypatch,
    settings_model,
):
    """Build a ProcessorSpec in DLCLiveWorker and clean it on reset."""
    from dlclivegui.services import dlc_processor as dlc_processor_module

    built_processors = []
    dlclive_instances = []

    class WorkerBuiltProcessor:
        PROCESSOR_NAME = "Worker-built test processor"

        def __init__(self):
            self.build_thread_name = threading.current_thread().name
            self.stop_calls = 0
            built_processors.append(self)

        def process(self, pose, **kwargs):
            return pose

        def stop(self):
            self.stop_calls += 1

    class FakeRunner:
        def get_pose(self, _processed_frame):
            return np.array(
                [[1.0, 2.0, 0.9]],
                dtype=np.float32,
            )

    class FakeDLCLive:
        def __init__(self, **options):
            self.processor = options["processor"]
            self.runner = FakeRunner()
            self.pose = None
            self.cfg = {}
            self.init_called = False
            dlclive_instances.append(self)

        def init_inference(self, _frame):
            self.init_called = True

        def process_frame(self, frame):
            return frame

    monkeypatch.setattr(
        dlc_processor_module,
        "DLCLive",
        FakeDLCLive,
    )

    proc = DLCLiveProcessor()
    proc.configure(
        settings_model,
        processor_spec=ProcessorSpec(
            cls=WorkerBuiltProcessor,
        ),
    )

    frame = np.zeros(
        (32, 32, 3),
        dtype=np.uint8,
    )

    try:
        with qtbot.waitSignal(
            proc.initialized,
            timeout=1500,
        ) as blocker:
            proc.enqueue_frame(frame, timestamp=1.0)

        assert blocker.args == [True]
        assert len(built_processors) == 1
        assert len(dlclive_instances) == 1

        built_processor = built_processors[0]
        dlclive_instance = dlclive_instances[0]

        assert built_processor.build_thread_name == "DLCLiveWorker"
        assert proc._processor is built_processor
        assert proc._processor_built_from_spec is True
        assert dlclive_instance.processor is built_processor
        assert dlclive_instance.init_called is True

        proc.reset(reset_processor_plugin=True)

        assert built_processor.stop_calls == 1
        assert proc._processor is None
        assert proc._processor_built_from_spec is False
        assert proc._state == WorkerState.STOPPED

    finally:
        proc.shutdown()


@pytest.mark.unit
def test_worker_processes_frames(qtbot, monkeypatch_dlclive, settings_model):
    proc = DLCLiveProcessor()
    proc.configure(settings_model)

    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        # The first frame should initialize DLCLive (initialized -> True) and produce the first pose.
        with qtbot.waitSignal(proc.initialized, timeout=1500):
            proc.enqueue_frame(frame, timestamp=1.0)

        # Wait for init pose
        qtbot.waitUntil(lambda: proc.get_stats().frames_processed >= 1, timeout=1500)

        # Enqueue more frames; wait for at least one more pose
        for i in range(10):
            proc.enqueue_frame(frame, timestamp=2.0 + i)
            qtbot.wait(5)  # ms

        # NOTE @C-Achard The timeout here is intentionally large to account for potential
        # Qt event-loop and threading scheduling delays in CI environments.
        # This was previously flaky with a smaller timeout; increasing it should
        # keep the test stable.
        qtbot.waitUntil(lambda: proc.get_stats().frames_processed >= 3, timeout=3000)

    finally:
        proc.reset()


def test_enqueue_frame_drops_stale_when_queue_is_full(settings_model):
    proc = DLCLiveProcessor()
    proc.configure(settings_model)

    try:
        proc._queue = queue.Queue(maxsize=1)
        proc._worker_thread = _AliveThread()
        proc._state = WorkerState.RUNNING
        proc._stop_event.clear()

        frame1 = np.zeros((32, 32, 3), dtype=np.uint8)
        frame2 = np.ones((32, 32, 3), dtype=np.uint8)

        proc.enqueue_frame(frame1, 1.0)
        proc.enqueue_frame(frame2, 2.0)

        stats = proc.get_stats()
        assert stats.frames_enqueued == 2
        assert stats.frames_dropped == 1
        assert stats.queue_size == 1

        queued_frame, queued_timestamp, _queued_at = proc._queue.get_nowait()
        assert queued_timestamp == 2.0
        np.testing.assert_array_equal(queued_frame, frame2)

    finally:
        proc._queue = None
        proc._worker_thread = None
        proc._state = WorkerState.STOPPED
        proc._stop_event.clear()


@pytest.mark.unit
def test_error_signal_on_initialization_failure(qtbot, monkeypatch):
    """Simulate DLCLive raising on init."""

    class FailingDLCLive:
        def __init__(self, **opts):
            raise RuntimeError("bad model")

    from dlclivegui.services import dlc_processor

    monkeypatch.setattr(dlc_processor, "DLCLive", FailingDLCLive)

    proc = DLCLiveProcessor()
    proc.configure(DLCProcessorSettings(model_path="fail.pt"))

    try:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        error_args = []
        init_args = []

        proc.error.connect(lambda msg: error_args.append(msg))
        proc.initialized.connect(lambda ok: init_args.append(ok))

        with qtbot.waitSignals([proc.error, proc.initialized], timeout=1500):
            proc.enqueue_frame(frame, 1.0)

        assert len(error_args) == 1
        assert "bad model" in error_args[0]

        assert len(init_args) == 1
        assert init_args[0] is False

    finally:
        proc.reset()


@pytest.mark.unit
def test_stats_computation(qtbot, monkeypatch_dlclive, settings_model):
    proc = DLCLiveProcessor()
    proc.configure(settings_model)

    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        # Start and wait for init
        with qtbot.waitSignal(proc.initialized, timeout=1500):
            proc.enqueue_frame(frame, 1.0)

        # Wait for init pose
        qtbot.waitUntil(lambda: proc.get_stats().frames_processed >= 1, timeout=1500)

        # Enqueue a second frame and wait for its pose
        proc.enqueue_frame(frame, 2.0)
        qtbot.waitUntil(lambda: proc.get_stats().frames_processed >= 2, timeout=1500)

        stats = proc.get_stats()
        assert isinstance(stats, ProcessorStats)
        assert stats.frames_processed >= 1
        assert stats.processing_fps >= 0

    finally:
        proc.reset()


@pytest.mark.unit
def test_worker_processes_second_frame_and_updates_stats(qtbot, monkeypatch_dlclive, settings_model):
    """
    Explicitly verify that after initialization, a queued frame is processed:
    - frame_processed is emitted for the second frame
    - frames_processed >= 2 (init + 1 queued)
    """
    proc = DLCLiveProcessor()
    proc.configure(settings_model)

    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        # First frame triggers initialization + init pose
        with qtbot.waitSignal(proc.initialized, timeout=1500):
            proc.enqueue_frame(frame, 1.0)
        qtbot.waitUntil(lambda: proc.get_stats().frames_processed >= 1, timeout=1500)  # init pose

        # Enqueue one more frame and wait for its pose
        proc.enqueue_frame(frame, 2.0)
        qtbot.waitUntil(lambda: proc.get_stats().frames_processed >= 2, timeout=1500)
        stats = proc.get_stats()
        # >= 2: init + the second frame
        assert stats.frames_processed >= 2
        # queue drained
        assert stats.queue_size == 0

    finally:
        proc.reset()


@pytest.mark.unit
def test_worker_survives_empty_timeouts_then_processes_next(qtbot, monkeypatch_dlclive, settings_model):
    """
    Verify the worker doesn't exit after queue.Empty timeouts and still processes
    a subsequent enqueued frame (this asserts the loop continues running).
    """
    proc = DLCLiveProcessor()
    proc.configure(settings_model)

    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        # Initialize with first frame
        with qtbot.waitSignal(proc.initialized, timeout=1500):
            proc.enqueue_frame(frame, 1.0)
        qtbot.waitUntil(lambda: proc.get_stats().frames_processed >= 1, timeout=1500)  # init pose

        # Let the worker spin with an empty queue (several 0.1s timeouts)
        qtbot.wait(350)  # ~3-4 timeouts

        # The worker thread should still be alive
        assert proc._worker_thread is not None and proc._worker_thread.is_alive()

        # Enqueue another frame and ensure it is processed
        proc.enqueue_frame(frame, 2.0)
        qtbot.waitUntil(lambda: proc.get_stats().frames_processed >= 2, timeout=1500)

        stats = proc.get_stats()
        assert stats.frames_processed >= 2

    finally:
        proc.reset()


@pytest.mark.unit
def test_queue_accounting_clears_after_processed_frame(qtbot, monkeypatch_dlclive, settings_model):
    """
    After a queued frame is processed:
    - queue size returns to zero
    - unfinished task count (if accessible) is zero

    This implicitly validates correct task_done() usage for processed items.
    Note: the init frame is not queued, so we only check queued work accounting.
    """
    proc = DLCLiveProcessor()
    proc.configure(settings_model)

    try:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        # Initialize (no queue involvement for the init frame)
        with qtbot.waitSignal(proc.initialized, timeout=1500):
            proc.enqueue_frame(frame, 1.0)
        qtbot.waitUntil(lambda: proc.get_stats().frames_processed >= 1, timeout=1500)  # init pose

        # Enqueue one queued frame
        proc.enqueue_frame(frame, 2.0)
        qtbot.waitUntil(lambda: proc.get_stats().frames_processed >= 2, timeout=1500)

        # Queue should be drained
        q = proc._queue
        # It's allowed to be None if the worker shut down, but in normal run it should exist
        if q is not None:
            assert q.qsize() == 0
            # CPython exposes 'unfinished_tasks'; if present, it should be zero
            unfinished = getattr(q, "unfinished_tasks", 0)
            assert unfinished == 0

    finally:
        proc.reset()
