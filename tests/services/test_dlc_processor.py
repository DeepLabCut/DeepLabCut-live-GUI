import numpy as np
import pytest

from dlclivegui.config import DLCProcessorSettings
from dlclivegui.services.dlc_processor import (
    DLCLiveProcessor,
    ProcessorStats,
)

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_configure_accepts_dataclass(settings_dc, monkeypatch_dlclive):
    proc = DLCLiveProcessor()
    proc.configure(settings_dc)

    assert proc._settings.model_path == "dummy.pt"
    assert proc._processor is None


@pytest.mark.unit
def test_configure_accepts_pydantic(settings_model, monkeypatch_dlclive):
    proc = DLCLiveProcessor()
    proc.configure(settings_model)

    # Should have normalized to dataclass internally
    assert isinstance(proc._settings, DLCProcessorSettings)
    assert proc._settings.model_path == "dummy.pt"


@pytest.mark.unit
def test_worker_initializes_on_first_frame(qtbot, monkeypatch_dlclive, settings_dc):
    proc = DLCLiveProcessor()
    proc.configure(settings_dc)

    try:
        # First enqueued frame triggers worker start + initialization.
        with qtbot.waitSignal(proc.initialized, timeout=1500) as init_blocker:
            proc.enqueue_frame(np.zeros((100, 100, 3), dtype=np.uint8), timestamp=1.0)

        assert init_blocker.args == [True]
        assert proc._initialized
        assert getattr(proc._dlc, "init_called", False)

        # Optional: also ensure the init pose was delivered
        qtbot.waitSignal(proc.pose_ready, timeout=1500)

    finally:
        proc.reset()  # Ensure thread cleanup


@pytest.mark.unit
def test_worker_processes_frames(qtbot, monkeypatch_dlclive, settings_dc):
    proc = DLCLiveProcessor()
    proc.configure(settings_dc)

    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        # The first frame should initialize DLCLive (initialized -> True) and produce the first pose.
        with qtbot.waitSignal(proc.initialized, timeout=1500):
            proc.enqueue_frame(frame, timestamp=1.0)

        # Wait for init pose
        qtbot.waitSignal(proc.pose_ready, timeout=1500)

        # Enqueue more frames; wait for at least one more pose
        for i in range(3):
            proc.enqueue_frame(frame, timestamp=2.0 + i)

        qtbot.waitSignal(proc.pose_ready, timeout=1500)

        assert proc._frames_processed >= 2  # at least init + one more

    finally:
        proc.reset()


@pytest.mark.unit
def test_queue_full_drops_frames(qtbot, monkeypatch_dlclive, settings_dc):
    proc = DLCLiveProcessor()
    proc.configure(settings_dc)

    try:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        # Start the worker with the first frame
        with qtbot.waitSignal(proc.initialized, timeout=1500):
            proc.enqueue_frame(frame, 1.0)

        # Flood the 1-slot queue to force drops
        for _ in range(50):
            proc.enqueue_frame(frame, 2.0)

        # Wait until we observe dropped frames
        qtbot.waitUntil(lambda: proc._frames_dropped > 0, timeout=1500)
        assert proc._frames_dropped > 0

    finally:
        proc.reset()


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
def test_stats_computation(qtbot, monkeypatch_dlclive, settings_dc):
    proc = DLCLiveProcessor()
    proc.configure(settings_dc)

    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        # Start and wait for init
        with qtbot.waitSignal(proc.initialized, timeout=1500):
            proc.enqueue_frame(frame, 1.0)

        # Wait for init pose
        qtbot.waitSignal(proc.pose_ready, timeout=1500)

        # Enqueue a second frame and wait for its pose
        proc.enqueue_frame(frame, 2.0)
        qtbot.waitSignal(proc.pose_ready, timeout=1500)

        stats = proc.get_stats()
        assert isinstance(stats, ProcessorStats)
        assert stats.frames_processed >= 1
        assert stats.processing_fps >= 0

    finally:
        proc.reset()
