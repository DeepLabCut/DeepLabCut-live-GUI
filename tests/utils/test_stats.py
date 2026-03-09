from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dlclivegui.utils.stats import format_dlc_stats, format_recorder_stats

pytestmark = pytest.mark.unit

# -----------------------------
# Exact formatting tests
# -----------------------------


def test_format_recorder_stats_exact():
    stats = SimpleNamespace(
        frames_written=10,
        frames_enqueued=12,
        write_fps=29.94,
        last_latency=0.01234,  # 12.34 ms -> 12.3
        average_latency=0.05678,  # 56.78 ms -> 56.8
        buffer_seconds=0.4321,  # 432.1 ms -> 432
        queue_size=3,
        dropped_frames=2,
    )

    assert format_recorder_stats(stats) == (
        "10/12 frames | write 29.9 fps | latency 12.3 ms (avg 56.8 ms) | queue 3 (~432 ms) | dropped 2"
    )


def test_format_dlc_stats_exact_no_profile():
    stats = SimpleNamespace(
        frames_processed=100,
        frames_enqueued=120,
        processing_fps=87.65,  # -> 87.7
        last_latency=0.001,  # 1.0 ms
        average_latency=0.00234,  # 2.3 ms
        queue_size=5,
        frames_dropped=7,
        # Profile fields disabled by avg_inference_time == 0
        avg_inference_time=0.0,
        avg_queue_wait=0.0,
        avg_signal_emit_time=0.0,
        avg_total_process_time=0.0,
        avg_gpu_inference_time=0.0,
        avg_processor_overhead=0.0,
    )

    assert format_dlc_stats(stats) == (
        "100/120 frames | inference 87.7 fps | latency 1.0 ms (avg 2.3 ms) | queue 5 | dropped 7"
    )


def test_format_dlc_stats_exact_with_profile_and_gpu_breakdown():
    stats = SimpleNamespace(
        frames_processed=3,
        frames_enqueued=4,
        processing_fps=12.34,  # -> 12.3
        last_latency=0.01001,  # 10.01 ms -> 10.0
        average_latency=0.02006,  # 20.06 ms -> 20.1
        queue_size=2,
        frames_dropped=1,
        # Profile enabled
        avg_inference_time=0.00555,  # 5.55 ms -> 5.5
        avg_queue_wait=0.00123,  # 1.23 ms -> 1.2
        avg_signal_emit_time=0.00049,  # 0.49 ms -> 0.5
        avg_total_process_time=0.00777,  # 7.77 ms -> 7.8
        # GPU breakdown enabled
        avg_gpu_inference_time=0.0008,  # 0.8 ms
        avg_processor_overhead=0.0002,  # 0.2 ms
    )

    assert format_dlc_stats(stats) == (
        "3/4 frames | "
        "inference 12.3 fps | "
        "latency 10.0 ms (avg 20.1 ms) | "
        "queue 2 | dropped 1"
        "\n[Profile] inf:5.5ms (GPU:0.8ms+proc:0.2ms) "
        "queue:1.2ms signal:0.5ms total:7.8ms"
    )


# -----------------------------
# Strategies (bounded & finite)
# -----------------------------
finite_seconds = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
finite_seconds_small = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
finite_fps = st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False)

nonneg_int = st.integers(min_value=0, max_value=1_000_000)
queue_size_int = st.integers(min_value=0, max_value=10_000)


def _fmt1(x: float) -> str:
    """Exactly the same rounding as f'{x:.1f}'."""
    return f"{x:.1f}"


def _fmt0(x: float) -> str:
    """Exactly the same rounding as f'{x:.0f}'."""
    return f"{x:.0f}"


# -----------------------------
# Recorder stats properties
# -----------------------------
@settings(max_examples=200, deadline=None)
@given(
    frames_written=nonneg_int,
    frames_enqueued=nonneg_int,
    write_fps=finite_fps,
    last_latency=finite_seconds_small,
    average_latency=finite_seconds_small,
    buffer_seconds=finite_seconds,
    queue_size=queue_size_int,
    dropped_frames=nonneg_int,
)
def test_format_recorder_stats_properties(
    frames_written,
    frames_enqueued,
    write_fps,
    last_latency,
    average_latency,
    buffer_seconds,
    queue_size,
    dropped_frames,
):
    stats = SimpleNamespace(
        frames_written=frames_written,
        frames_enqueued=frames_enqueued,
        write_fps=write_fps,
        last_latency=last_latency,
        average_latency=average_latency,
        buffer_seconds=buffer_seconds,
        queue_size=queue_size,
        dropped_frames=dropped_frames,
    )

    s = format_recorder_stats(stats)

    # Required structural tokens
    assert " frames | write " in s
    assert " fps | latency " in s
    assert " ms (avg " in s
    assert " ms) | queue " in s
    assert " (~" in s
    assert " ms) | dropped " in s

    # Exact numeric formatting expectations (substrings)
    latency_ms = last_latency * 1000.0
    avg_ms = average_latency * 1000.0
    buffer_ms = buffer_seconds * 1000.0

    assert f"{frames_written}/{frames_enqueued} frames" in s
    assert f"write {_fmt1(write_fps)} fps" in s
    assert f"latency {_fmt1(latency_ms)} ms (avg {_fmt1(avg_ms)} ms)" in s
    assert f"queue {queue_size} (~{_fmt0(buffer_ms)} ms)" in s
    assert f"dropped {dropped_frames}" in s


# -----------------------------
# DLC stats properties
# -----------------------------
def dlc_stats_strategy(profile_enabled: bool):
    """
    Build a strategy for DLC stats where profile block is enabled/disabled.
    - profile enabled iff avg_inference_time > 0
    """
    if profile_enabled:
        avg_inf = st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False)
    else:
        avg_inf = st.floats(min_value=0.0, max_value=0.0, allow_nan=False, allow_infinity=False)

    # For profile enabled case, allow sub-times to be 0..1s
    avg_sub = finite_seconds_small

    return st.fixed_dictionaries(
        {
            "frames_processed": nonneg_int,
            "frames_enqueued": nonneg_int,
            "processing_fps": finite_fps,
            "last_latency": finite_seconds_small,
            "average_latency": finite_seconds_small,
            "queue_size": queue_size_int,
            "frames_dropped": nonneg_int,
            "avg_inference_time": avg_inf,
            "avg_queue_wait": avg_sub,
            "avg_signal_emit_time": avg_sub,
            "avg_total_process_time": avg_sub,
            "avg_gpu_inference_time": avg_sub,
            "avg_processor_overhead": avg_sub,
        }
    ).map(lambda d: SimpleNamespace(**d))


@settings(max_examples=200, deadline=None)
@given(stats=dlc_stats_strategy(profile_enabled=False))
def test_format_dlc_stats_no_profile_properties(stats):
    s = format_dlc_stats(stats)

    # Core structure always present
    assert f"{stats.frames_processed}/{stats.frames_enqueued} frames" in s
    assert f"inference {_fmt1(stats.processing_fps)} fps" in s

    latency_ms = stats.last_latency * 1000.0
    avg_ms = stats.average_latency * 1000.0
    assert f"latency {_fmt1(latency_ms)} ms (avg {_fmt1(avg_ms)} ms)" in s

    assert f"queue {stats.queue_size} | dropped {stats.frames_dropped}" in s

    # Profile must NOT be present
    assert "\n[Profile]" not in s
    assert "GPU:" not in s


@settings(max_examples=250, deadline=None)
@given(stats=dlc_stats_strategy(profile_enabled=True))
def test_format_dlc_stats_profile_properties(stats):
    s = format_dlc_stats(stats)

    # Core structure
    assert f"{stats.frames_processed}/{stats.frames_enqueued} frames" in s
    assert f"inference {_fmt1(stats.processing_fps)} fps" in s

    latency_ms = stats.last_latency * 1000.0
    avg_ms = stats.average_latency * 1000.0
    assert f"latency {_fmt1(latency_ms)} ms (avg {_fmt1(avg_ms)} ms)" in s

    assert f"queue {stats.queue_size} | dropped {stats.frames_dropped}" in s

    # Profile must be present
    assert "\n[Profile]" in s

    inf_ms = stats.avg_inference_time * 1000.0
    queue_ms = stats.avg_queue_wait * 1000.0
    signal_ms = stats.avg_signal_emit_time * 1000.0
    total_ms = stats.avg_total_process_time * 1000.0

    assert f"inf:{_fmt1(inf_ms)}ms" in s
    assert f"queue:{_fmt1(queue_ms)}ms" in s
    assert f"signal:{_fmt1(signal_ms)}ms" in s
    assert f"total:{_fmt1(total_ms)}ms" in s

    # GPU breakdown is conditional:
    gpu_on = (stats.avg_gpu_inference_time > 0) or (stats.avg_processor_overhead > 0)
    if gpu_on:
        gpu_ms = stats.avg_gpu_inference_time * 1000.0
        proc_ms = stats.avg_processor_overhead * 1000.0
        assert f"(GPU:{_fmt1(gpu_ms)}ms+proc:{_fmt1(proc_ms)}ms)" in s
    else:
        assert "GPU:" not in s
