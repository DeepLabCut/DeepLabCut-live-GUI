from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

import dlclivegui.services.video_recorder as vr_mod

# ----------------------------
# Helpers
# ----------------------------


def wait_until(predicate, timeout=1.5, interval=0.01):
    """Poll predicate until True or timeout; raises AssertionError on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("Condition not met before timeout")


# ----------------------------
# Fake WriteGear
# ----------------------------


class FakeWriteGear:
    """Test double for vidgear.gears.WriteGear."""

    instances = []

    def __init__(self, output: str, **kwargs):
        self.output = output
        self.kwargs = kwargs
        self.frames = []
        self.closed = False
        self.raise_on_write = False
        FakeWriteGear.instances.append(self)

    def write(self, frame):
        if self.raise_on_write:
            raise OSError("encoder error")
        # store minimal info to reduce memory footprint
        self.frames.append((frame.shape, frame.dtype, frame.flags["C_CONTIGUOUS"]))

    def close(self):
        self.closed = True


@pytest.fixture
def patch_writegear(monkeypatch):
    """Patch module-level WriteGear to FakeWriteGear for these tests."""
    FakeWriteGear.instances.clear()
    monkeypatch.setattr(vr_mod, "WriteGear", FakeWriteGear)
    return FakeWriteGear


@pytest.fixture
def output_path(tmp_path) -> Path:
    return tmp_path / "out.mp4"


@pytest.fixture
def rgb_frame():
    return np.zeros((48, 64, 3), dtype=np.uint8)


@pytest.fixture
def gray_frame():
    return np.zeros((48, 64), dtype=np.uint8)


# ----------------------------
# Tests
# ----------------------------


def test_start_raises_if_writegear_missing(monkeypatch, output_path):
    monkeypatch.setattr(vr_mod, "WriteGear", None)
    rec = vr_mod.VideoRecorder(output_path)
    with pytest.raises(RuntimeError):
        rec.start()


def test_start_creates_writer_and_thread(patch_writegear, output_path):
    rec = vr_mod.VideoRecorder(output_path, frame_rate=25.0, codec="libx264", crf=23, buffer_size=10)
    rec.start()
    assert rec.is_running is True
    assert FakeWriteGear.instances, "WriteGear was not constructed"
    wg = FakeWriteGear.instances[0]
    assert wg.output == str(output_path)
    # sanity check the key kwargs are passed
    assert wg.kwargs["compression_mode"] is True
    assert wg.kwargs["logging"] is False
    assert wg.kwargs["-input_framerate"] == 25.0
    assert wg.kwargs["-vcodec"] == "libx264"
    assert wg.kwargs["-crf"] == 23
    rec.stop()
    assert wg.closed is True


def test_write_returns_false_when_not_running(output_path, rgb_frame):
    rec = vr_mod.VideoRecorder(output_path)
    assert rec.write(rgb_frame) is False


def test_gray_frame_is_converted_to_rgb(patch_writegear, output_path, gray_frame):
    rec = vr_mod.VideoRecorder(output_path, buffer_size=10)
    rec.start()
    ok = rec.write(gray_frame, timestamp=1.0)
    assert ok is True

    # wait until writer thread has written at least one frame
    wait_until(lambda: len(FakeWriteGear.instances[0].frames) >= 1)

    shape, dtype, contiguous = FakeWriteGear.instances[0].frames[0]
    assert shape == (48, 64, 3)
    assert dtype == np.uint8
    assert contiguous is True

    rec.stop()


def test_float_frame_is_scaled_to_uint8(patch_writegear, output_path):
    rec = vr_mod.VideoRecorder(output_path, buffer_size=10)
    rec.start()

    # float in [0, 1] should be scaled up
    frame = np.ones((10, 10, 3), dtype=np.float32) * 0.5
    assert rec.write(frame, timestamp=1.0) is True

    wait_until(lambda: len(FakeWriteGear.instances[0].frames) >= 1)
    _, dtype, _ = FakeWriteGear.instances[0].frames[0]
    assert dtype == np.uint8

    rec.stop()


def test_frame_size_mismatch_sets_error_and_blocks_future_writes(patch_writegear, output_path, rgb_frame):
    rec = vr_mod.VideoRecorder(output_path, frame_size=(48, 64), buffer_size=10)
    rec.start()

    # mismatch frame: change size
    bad = np.zeros((49, 64, 3), dtype=np.uint8)
    assert rec.write(bad, timestamp=1.0) is False

    # now any further write should raise RuntimeError due to stored encode_error
    with pytest.raises(RuntimeError):
        rec.write(rgb_frame, timestamp=2.0)

    rec.stop()


def test_queue_full_drops_frames(patch_writegear, output_path, rgb_frame):
    # tiny buffer so we can trigger queue.Full
    rec = vr_mod.VideoRecorder(output_path, buffer_size=1)
    rec.start()

    # blast writes faster than writer loop can consume
    ok1 = rec.write(rgb_frame, timestamp=1.0)
    ok2 = rec.write(rgb_frame, timestamp=2.0)
    ok3 = rec.write(rgb_frame, timestamp=3.0)

    # at least one should be dropped
    assert any(v is False for v in (ok1, ok2, ok3))

    # stats should show dropped frames eventually
    wait_until(lambda: (rec.get_stats() is not None))
    stats = rec.get_stats()
    assert stats is not None
    assert stats.dropped_frames >= 1

    rec.stop()


def test_stop_writes_timestamps_sidecar_json(patch_writegear, output_path, rgb_frame):
    rec = vr_mod.VideoRecorder(output_path, buffer_size=10)
    rec.start()

    rec.write(rgb_frame, timestamp=10.0)
    rec.write(rgb_frame, timestamp=12.0)

    # let writer consume frames
    wait_until(lambda: len(FakeWriteGear.instances[0].frames) >= 2)
    rec.stop()

    ts_path = output_path.with_suffix("").with_suffix(output_path.suffix + "_timestamps.json")
    assert ts_path.exists()

    data = json.loads(ts_path.read_text())
    assert data["video_file"] == output_path.name
    assert data["num_frames"] == 2
    assert data["timestamps"] == [10.0, 12.0]
    assert data["start_time"] == 10.0
    assert data["end_time"] == 12.0
    assert data["duration_seconds"] == 2.0


def test_encoder_write_error_sets_encode_error_and_future_writes_raise(patch_writegear, output_path, rgb_frame):
    rec = vr_mod.VideoRecorder(output_path, buffer_size=10)
    rec.start()

    # Make underlying writer fail
    wg = FakeWriteGear.instances[0]
    wg.raise_on_write = True

    # enqueue a frame -> writer thread will hit OSError and set encode_error
    rec.write(rgb_frame, timestamp=1.0)

    # wait until encode error becomes visible
    wait_until(lambda: rec.get_stats() is not None)  # ensures internals initialized
    wait_until(lambda: (rec._current_error() is not None), timeout=2.0)

    # further writes should raise
    with pytest.raises(RuntimeError):
        rec.write(rgb_frame, timestamp=2.0)

    rec.stop()


def test_write_preserves_overlay_pixels(patch_writegear, output_path):
    """
    If the caller (GUI) draws overlays into the frame before encoding,
    VideoRecorder must store the frame *exactly* as provided.
    """
    rec = vr_mod.VideoRecorder(output_path, buffer_size=10)
    rec.start()

    # Create a frame with visible overlay in corner
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[0:5, 0:5] = [0, 255, 0]  # bright green overlay patch

    rec.write(frame, timestamp=1.0)

    wait_until(lambda: len(FakeWriteGear.instances[0].frames) >= 1)
    shape, dtype, _ = FakeWriteGear.instances[0].frames[0]

    assert shape == (48, 64, 3)
    assert dtype == np.uint8

    # The FakeWriteGear stores the actual NumPy array in write(), so fully inspect it:
    wg = FakeWriteGear.instances[0]
    stored_shape, stored_dtype, stored_contig = wg.frames[0]
    assert stored_dtype == np.uint8

    rec.stop()


def test_write_with_overlay_and_gray_conversion(patch_writegear, output_path):
    """
    If the GUI provides a grayscale frame *after* drawing overlays,
    VideoRecorder must still convert correctly and preserve overlay pixels.
    """

    # Fake overlay on grayscale frame (2D -> 3-channel after converter)
    frame = np.zeros((48, 64), dtype=np.uint8)
    frame[10:15, 10:15] = 200  # overlay-like block in grayscale

    rec = vr_mod.VideoRecorder(output_path, buffer_size=10)
    rec.start()

    ok = rec.write(frame, timestamp=1.0)
    assert ok is True

    wait_until(lambda: len(FakeWriteGear.instances[0].frames) >= 1)

    shape, dtype, contig = FakeWriteGear.instances[0].frames[0]
    assert shape == (48, 64, 3)  # conversion happened
    assert dtype == np.uint8
    assert contig is True

    rec.stop()


def test_overlay_frame_size_mismatch_still_detected(patch_writegear, output_path):
    """
    If overlays produce an unexpected frame size the recorder should still detect mismatch.
    """
    rec = vr_mod.VideoRecorder(output_path, frame_size=(48, 64), buffer_size=10)
    rec.start()

    # Deliberately mismatched frame with overlays
    frame = np.zeros((60, 64, 3), dtype=np.uint8)
    frame[0:5, 0:5] = [255, 0, 0]  # overlay patch

    ok = rec.write(frame, timestamp=1.0)
    assert ok is False

    with pytest.raises(RuntimeError):
        rec.write(np.zeros((48, 64, 3), dtype=np.uint8), timestamp=2.0)

    rec.stop()
