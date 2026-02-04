from pathlib import Path

import pytest

import dlclivegui.utils.utils as u

pytestmark = pytest.mark.unit


# -----------------------------
# is_model_file
# -----------------------------
@pytest.mark.unit
def test_is_model_file_true_for_supported_extensions(tmp_path: Path):
    for ext in [".pt", ".pth", ".pb"]:
        p = tmp_path / f"model{ext}"
        p.write_text("x")
        assert u.is_model_file(p) is True
        assert u.is_model_file(str(p)) is True  # also accepts str

    # case-insensitive
    p2 = tmp_path / "MODEL.PT"
    p2.write_text("x")
    assert u.is_model_file(p2) is True


@pytest.mark.unit
def test_is_model_file_false_for_missing_or_dir(tmp_path: Path):
    missing = tmp_path / "missing.pt"
    assert u.is_model_file(missing) is False

    d = tmp_path / "model.pt"
    d.mkdir()
    assert u.is_model_file(d) is False

    bad = tmp_path / "model.onnx"
    bad.write_text("x")
    assert u.is_model_file(bad) is False


# -----------------------------
# sanitize_name
# -----------------------------
@pytest.mark.unit
def test_sanitize_name_fallback_and_trimming():
    assert u.sanitize_name("") == "session"
    assert u.sanitize_name("   ") == "session"
    assert u.sanitize_name(None) == "session"  # type: ignore[arg-type]
    assert u.sanitize_name("", fallback="x") == "x"

    # Strips leading/trailing punctuation and spaces
    assert u.sanitize_name("  ..__hello--  ") == "hello"


@pytest.mark.unit
def test_sanitize_name_replaces_invalid_chars():
    # invalid -> underscore; allowed: A-Za-z0-9._-
    assert u.sanitize_name("my session!") == "my_session"
    assert u.sanitize_name("a/b\\c:d*e?f") == "a_b_c_d_e_f"
    # collapse behavior is regex-based, not explicitly collapsing multiple underscores,
    # so we only assert it's "safe" and non-empty.
    out = u.sanitize_name("###")
    assert out == "session"


# -----------------------------
# timestamp_string
# -----------------------------
@pytest.mark.unit
def test_timestamp_string_formats(monkeypatch):
    # Patch the module's imported 'datetime' symbol (from datetime import datetime)
    class FakeDateTime:
        @staticmethod
        def now():
            # 2026-02-04 15:20:18.123456 -> with_ms => "20260204_152018_123"
            from datetime import datetime as _dt

            return _dt(2026, 2, 4, 15, 20, 18, 123456)

    monkeypatch.setattr(u, "datetime", FakeDateTime)

    assert u.timestamp_string(with_ms=True) == "20260204_152018_123"
    assert u.timestamp_string(with_ms=False) == "20260204_152018"


# -----------------------------
# split_stem_ext
# -----------------------------
@pytest.mark.unit
def test_split_stem_ext_keeps_user_extension():
    stem, ext = u.split_stem_ext("video.avi", "mp4")
    assert stem == "video"
    assert ext == "avi"


@pytest.mark.unit
def test_split_stem_ext_uses_container_when_no_extension():
    stem, ext = u.split_stem_ext("video", "mp4")
    assert stem == "video"
    assert ext == "mp4"

    stem, ext = u.split_stem_ext("video", ".mov")
    assert stem == "video"
    assert ext == "mov"


@pytest.mark.unit
def test_split_stem_ext_defaults_when_empty():
    stem, ext = u.split_stem_ext("", "")
    assert stem == "recording"
    assert ext == "mp4"


# -----------------------------
# next_run_index
# -----------------------------
@pytest.mark.unit
def test_next_run_index_finds_next_numeric_dir(tmp_path: Path):
    # ignores files, non-matching dirs, non-digit suffixes
    (tmp_path / "run_0001").mkdir()
    (tmp_path / "run_0003").mkdir()
    (tmp_path / "run_foo").mkdir()
    (tmp_path / "notrun_0009").mkdir()
    (tmp_path / "run_0002").write_text("file-not-dir")

    assert u.next_run_index(tmp_path) == 4


@pytest.mark.unit
def test_next_run_index_custom_prefix(tmp_path: Path):
    (tmp_path / "take_0002").mkdir()
    (tmp_path / "take_0005").mkdir()
    assert u.next_run_index(tmp_path, prefix="take_") == 6


# -----------------------------
# build_run_dir
# -----------------------------
@pytest.mark.unit
def test_build_run_dir_incrementing(tmp_path: Path):
    session_dir = tmp_path / "session"
    # First time -> run_0001
    rd1 = u.build_run_dir(session_dir, use_timestamp=False)
    assert rd1.name == "run_0001"
    assert rd1.exists() and rd1.is_dir()

    # Second time -> run_0002
    rd2 = u.build_run_dir(session_dir, use_timestamp=False)
    assert rd2.name == "run_0002"
    assert rd2.exists() and rd2.is_dir()


@pytest.mark.unit
def test_build_run_dir_timestamp_unique_on_collision(tmp_path: Path, monkeypatch):
    session_dir = tmp_path / "session"

    # Make timestamp_string return a constant, then a second value to resolve collision
    stamps = iter(["20260204_152018_123", "20260204_152018_999"])

    def fake_timestamp_string(*, with_ms=True):
        return next(stamps)

    monkeypatch.setattr(u, "timestamp_string", fake_timestamp_string)

    # Pre-create the first "collision" directory
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "run_20260204_152018_123").mkdir()

    rd = u.build_run_dir(session_dir, use_timestamp=True)
    assert rd.name == "run_20260204_152018_123_20260204_152018_999"
    assert rd.exists() and rd.is_dir()


# -----------------------------
# build_recording_plan
# -----------------------------
@pytest.mark.unit
def test_build_recording_plan_sanitizes_names_and_builds_paths(tmp_path: Path, monkeypatch):
    # Make run_dir deterministic using incrementing mode, and stable timestamp not needed
    output_dir = tmp_path / "out"
    camera_ids = ["cam:0", "Left Cam", "###"]

    plan = u.build_recording_plan(
        output_dir=output_dir,
        session_name="  My Session!!  ",
        base_filename="  base name  ",  # no extension
        container=".mp4",
        camera_ids=camera_ids,
        use_timestamp=False,
    )

    # session dir uses sanitize_name
    assert plan.session_dir == output_dir / "My_Session"
    assert plan.session_dir.exists()

    # run dir incrementing
    assert plan.run_dir.name == "run_0001"
    assert plan.run_dir.exists()

    # file naming: {stem}_{safe_cam}.{ext}
    # stem sanitize: "base name" -> "base_name"
    # cam id sanitize: cam:0 -> cam_0 (':' replaced then sanitized)
    assert plan.files_by_camera_id["cam:0"].name == "base_name_cam_0.mp4"
    assert plan.files_by_camera_id["Left Cam"].name == "base_name_Left_Cam.mp4"
    # "###" becomes fallback "cam"
    assert plan.files_by_camera_id["###"].name == "base_name_cam.mp4"

    # ensure all are under run_dir
    for _cid, path in plan.files_by_camera_id.items():
        assert path.parent == plan.run_dir


@pytest.mark.unit
def test_build_recording_plan_respects_user_extension(tmp_path: Path):
    plan = u.build_recording_plan(
        output_dir=tmp_path,
        session_name="s",
        base_filename="video.avi",
        container="mp4",  # should be ignored because base_filename already has ext
        camera_ids=["cam1"],
        use_timestamp=False,
    )
    assert plan.files_by_camera_id["cam1"].name.endswith(".avi")


# -----------------------------
# FPSTracker
# -----------------------------
@pytest.mark.unit
def test_fps_tracker_returns_zero_until_two_frames(monkeypatch):
    # Patch perf_counter for deterministic timestamps
    t = iter([1.0, 1.1])  # only one note_frame call will consume 1.0
    monkeypatch.setattr(u.time, "perf_counter", lambda: next(t))

    tr = u.FPSTracker(window_seconds=5.0)
    tr.note_frame("cam")
    assert tr.fps("cam") == 0.0  # fewer than 2 frames -> 0


@pytest.mark.unit
def test_fps_tracker_computes_fps(monkeypatch):
    # 4 frames at times: 0.0, 0.5, 1.0, 1.5 => (4-1)/(1.5-0.0) = 2.0 fps
    times = iter([0.0, 0.5, 1.0, 1.5])
    monkeypatch.setattr(u.time, "perf_counter", lambda: next(times))

    tr = u.FPSTracker(window_seconds=5.0)
    for _ in range(4):
        tr.note_frame("cam")

    assert tr.fps("cam") == pytest.approx(2.0, rel=1e-6)


@pytest.mark.unit
def test_fps_tracker_window_eviction(monkeypatch):
    # Window 1.0s: keep only frames within last 1.0 second
    # times: 0.0, 0.4, 0.8, 1.2 -> when at 1.2, frames older than 0.2 evicted => 0.4,0.8,1.2 remain
    # fps = (3-1)/(1.2-0.4) = 2.5 fps
    times = iter([0.0, 0.4, 0.8, 1.2])
    monkeypatch.setattr(u.time, "perf_counter", lambda: next(times))

    tr = u.FPSTracker(window_seconds=1.0)
    for _ in range(4):
        tr.note_frame("cam")

    assert tr.fps("cam") == pytest.approx(2.5, rel=1e-6)


@pytest.mark.unit
def test_fps_tracker_clear(monkeypatch):
    times = iter([0.0, 0.5])
    monkeypatch.setattr(u.time, "perf_counter", lambda: next(times))

    tr = u.FPSTracker(window_seconds=5.0)
    tr.note_frame("cam")
    tr.note_frame("cam")
    assert tr.fps("cam") > 0

    tr.clear()
    assert tr.fps("cam") == 0.0
