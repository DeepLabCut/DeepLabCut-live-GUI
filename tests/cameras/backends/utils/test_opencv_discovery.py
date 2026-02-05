# tests/test_opencv_discovery.py
from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

MODULE_PATH = "dlclivegui.cameras.backends.utils.opencv_discovery"


def _ensure_stub_cv2(monkeypatch):
    """
    Ensure importing opencv_discovery.py does not require OpenCV.
    If real cv2 exists, we leave it alone.
    Otherwise we inject a minimal stub into sys.modules.
    """
    if "cv2" in sys.modules:
        return

    stub = SimpleNamespace(
        CAP_ANY=0,
        CAP_DSHOW=700,
        CAP_MSMF=1400,
        CAP_AVFOUNDATION=1200,
        CAP_V4L2=200,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        VideoCapture=object,  # not used directly in tests (we patch try_open)
    )
    monkeypatch.setitem(sys.modules, "cv2", stub)


@pytest.fixture()
def od(monkeypatch):
    """
    Import the module under test, ensuring cv2 is available (real or stub).
    Then return the imported module object.
    """
    _ensure_stub_cv2(monkeypatch)
    # Reload to avoid cross-test contamination when monkeypatching module globals
    mod = importlib.import_module(MODULE_PATH)
    importlib.reload(mod)
    return mod


# ----------------------------
# Basic math helpers
# ----------------------------


def test_aspect_and_aspect_close(od):
    assert od._aspect(4, 3) == pytest.approx(4 / 3)
    assert od._aspect(0, 3) == 0.0
    assert od._aspect(3, 0) == 0.0

    a = 4 / 3
    assert od._aspect_close(a, a, 0.0) is True
    assert od._aspect_close(a, a * 1.005, 0.01) is True  # within 1%
    assert od._aspect_close(a, a * 1.02, 0.01) is False  # outside 1%
    assert od._aspect_close(0.0, a, 0.01) is False
    assert od._aspect_close(a, 0.0, 0.01) is False


# ----------------------------
# CameraCandidate / selection
# ----------------------------


def test_camera_candidate_stable_id_prefers_path(od):
    c = od.CameraCandidate(index=1, backend=2, name="Cam", path="unique", vid=0x046D, pid=0x0825)
    assert c.stable_id == "path:unique"


def test_camera_candidate_stable_id_uses_vid_pid_when_no_path(od):
    c = od.CameraCandidate(index=1, backend=2, name="Cam", path="", vid=0x046D, pid=0x0825)
    assert c.stable_id.startswith("usb:046d:0825:Cam")


def test_camera_candidate_stable_id_fallback(od):
    c = od.CameraCandidate(index=7, backend=9, name="Cam", path="")
    assert c.stable_id == "name:Cam:idx:7:b:9"


def test_select_camera_priority_order(od):
    cams = [
        od.CameraCandidate(index=0, backend=1, name="Integrated", path="p0", vid=1, pid=1),
        od.CameraCandidate(index=1, backend=1, name="Logitech C920", path="", vid=0x046D, pid=0x082D),
        od.CameraCandidate(index=2, backend=1, name="Other", path="p2", vid=2, pid=2),
    ]

    # 1) stable_id exact
    chosen = od.select_camera(cams, prefer_stable_id="path:p2")
    assert chosen.index == 2

    # 2) VID/PID
    chosen = od.select_camera(cams, prefer_vid_pid=(0x046D, 0x082D))
    assert chosen.index == 1

    # 3) name substring
    chosen = od.select_camera(cams, prefer_name_substr="c920")
    assert chosen.index == 1

    # 4) fallback_index
    chosen = od.select_camera(cams, fallback_index=0)
    assert chosen.index == 0

    # 5) first
    chosen = od.select_camera(cams)
    assert chosen.index == 0


# ----------------------------
# list_cameras() behavior
# ----------------------------


def test_list_cameras_with_injected_enumerator(od):
    class Info:
        def __init__(self, index, backend, name, path, vid, pid):
            self.index = index
            self.backend = backend
            self.name = name
            self.path = path
            self.vid = vid
            self.pid = pid

    def fake_enum(api_pref):
        assert api_pref == 123
        return [
            Info(0, 700, "Cam0", "id0", 0x1111, 0x2222),
            Info(1, 700, "Cam1", "", None, None),
        ]

    cams = od.list_cameras(api_preference=123, enumerator=fake_enum)
    assert len(cams) == 2
    assert cams[0].index == 0
    assert cams[0].backend == 700
    assert cams[0].name == "Cam0"
    assert cams[0].path == "id0"
    assert cams[0].vid == 0x1111
    assert cams[0].pid == 0x2222


def test_list_cameras_handles_enumerator_exception(od, caplog):
    def broken_enum(_api_pref):
        raise RuntimeError("boom")

    caplog.set_level("DEBUG")
    cams = od.list_cameras(api_preference=0, enumerator=broken_enum)
    assert cams == []


# ----------------------------
# preferred backend selection
# ----------------------------


def test_preferred_backend_for_platform_windows(od, monkeypatch):
    monkeypatch.setattr(od.platform, "system", lambda: "Windows")
    assert od.preferred_backend_for_platform() == od.cv2.CAP_DSHOW


def test_preferred_backend_for_platform_macos(od, monkeypatch):
    monkeypatch.setattr(od.platform, "system", lambda: "Darwin")
    assert od.preferred_backend_for_platform() == od.cv2.CAP_AVFOUNDATION


def test_preferred_backend_for_platform_linux(od, monkeypatch):
    monkeypatch.setattr(od.platform, "system", lambda: "Linux")
    assert od.preferred_backend_for_platform() == od.cv2.CAP_V4L2


# ----------------------------
# open_with_fallbacks() behavior
# ----------------------------


def test_open_with_fallbacks_pref_backend_succeeds(od, monkeypatch):
    fake_cap = object()

    def fake_try_open(index, backend):
        # succeed on first call
        return fake_cap

    monkeypatch.setattr(od, "try_open", fake_try_open)
    monkeypatch.setattr(od.platform, "system", lambda: "Windows")

    cap, spec = od.open_with_fallbacks(0, od.cv2.CAP_DSHOW)
    assert cap is fake_cap
    assert spec.index == 0
    assert spec.used_fallback is False


def test_open_with_fallbacks_windows_falls_back_to_msmf(od, monkeypatch):
    fake_cap = object()

    def fake_try_open(index, backend):
        # fail for DSHOW, succeed for MSMF
        if backend == od.cv2.CAP_MSMF:
            return fake_cap
        return None

    monkeypatch.setattr(od, "try_open", fake_try_open)
    monkeypatch.setattr(od.platform, "system", lambda: "Windows")

    cap, spec = od.open_with_fallbacks(0, od.cv2.CAP_DSHOW)
    assert cap is fake_cap
    assert spec.backend == od.cv2.CAP_MSMF
    assert spec.used_fallback is True


def test_open_with_fallbacks_finally_any(od, monkeypatch):
    fake_cap = object()

    def fake_try_open(index, backend):
        # only succeed on ANY
        if backend == od.cv2.CAP_ANY:
            return fake_cap
        return None

    monkeypatch.setattr(od, "try_open", fake_try_open)
    monkeypatch.setattr(od.platform, "system", lambda: "Linux")

    cap, spec = od.open_with_fallbacks(0, od.cv2.CAP_V4L2)
    assert cap is fake_cap
    assert spec.backend == od.cv2.CAP_ANY
    assert spec.used_fallback is True


# ----------------------------
# candidate generation
# ----------------------------


def test_generate_candidates_contains_exact_and_dedup(od):
    cands = od.generate_candidates(1024, 768, "strict")
    assert (1024, 768) in cands
    # should be unique
    assert len(cands) == len(set(cands))


def test_generate_candidates_includes_4_3_standards_when_4_3(od):
    cands = od.generate_candidates(1024, 768, "strict")
    for wh in [(640, 480), (800, 600), (1024, 768), (1280, 960)]:
        assert wh in cands


def test_generate_candidates_includes_16_9_standards_when_16_9(od):
    cands = od.generate_candidates(1280, 720, "strict")
    for wh in [(640, 360), (960, 540), (1280, 720), (1920, 1080)]:
        assert wh in cands


# ----------------------------
# apply_mode_with_verification() with fake capture
# ----------------------------
def test_apply_mode_with_verification_accepts_exact_match_strict(od, monkeypatch, fake_capture_factory):
    # Patch CAP_PROP constants in module's cv2 (works with real or stub)
    monkeypatch.setattr(od.cv2, "CAP_PROP_FRAME_WIDTH", 3, raising=False)
    monkeypatch.setattr(od.cv2, "CAP_PROP_FRAME_HEIGHT", 4, raising=False)
    monkeypatch.setattr(od.cv2, "CAP_PROP_FPS", 5, raising=False)

    cap = fake_capture_factory(grant_map={(1024, 768): (1024, 768)}, fps=30.0)

    req = od.ModeRequest(width=1024, height=768, fps=30.0, enforce_aspect="strict")
    res = od.apply_mode_with_verification(cap, req, warmup_grabs=0)

    assert res.accepted is True
    assert (res.width, res.height) == (1024, 768)


def test_apply_mode_with_verification_strict_aspect_skips_wrong_aspect(od, monkeypatch, fake_capture_factory):
    monkeypatch.setattr(od.cv2, "CAP_PROP_FRAME_WIDTH", 3, raising=False)
    monkeypatch.setattr(od.cv2, "CAP_PROP_FRAME_HEIGHT", 4, raising=False)
    monkeypatch.setattr(od.cv2, "CAP_PROP_FPS", 5, raising=False)

    # If the camera "grants" 1280x720 (16:9) even when asked 1024x768, strict should not accept it.
    cap = fake_capture_factory(grant_map={(1024, 768): (1280, 720)}, fps=30.0)

    req = od.ModeRequest(width=1024, height=768, fps=30.0, enforce_aspect="strict")

    res = od.apply_mode_with_verification(
        cap,
        req,
        candidates=[(1024, 768)],
        warmup_grabs=0,
    )

    assert res.accepted is False
    assert (res.width, res.height) == (1280, 720)
    # but we still return the best attempt (so non-zero)
    assert res.width > 0 and res.height > 0


def test_apply_mode_with_verification_strict_aspect_can_choose_alternative_aspect_preserving_mode(
    od, monkeypatch, fake_capture_factory
):
    monkeypatch.setattr(od.cv2, "CAP_PROP_FRAME_WIDTH", 3, raising=False)
    monkeypatch.setattr(od.cv2, "CAP_PROP_FRAME_HEIGHT", 4, raising=False)
    monkeypatch.setattr(od.cv2, "CAP_PROP_FPS", 5, raising=False)

    # Only the exact request is wrong-aspect; other candidates will be granted as requested
    cap = fake_capture_factory(grant_map={(1024, 768): (1280, 720)}, fps=30.0)

    req = od.ModeRequest(width=1024, height=768, fps=30.0, enforce_aspect="strict")
    res = od.apply_mode_with_verification(cap, req, warmup_grabs=0)

    assert res.accepted is True
    assert abs((res.width / res.height) - (4 / 3)) < 0.02


def test_apply_mode_with_verification_returns_best_when_none_accepted(od, monkeypatch, fake_capture_factory):
    monkeypatch.setattr(od.cv2, "CAP_PROP_FRAME_WIDTH", 3, raising=False)
    monkeypatch.setattr(od.cv2, "CAP_PROP_FRAME_HEIGHT", 4, raising=False)
    monkeypatch.setattr(od.cv2, "CAP_PROP_FPS", 5, raising=False)

    # Always map to a "bad" aspect
    cap = fake_capture_factory(grant_map={(640, 480): (1280, 720)}, fps=30.0)

    # Provide candidates to keep the test bounded/deterministic
    req = od.ModeRequest(width=640, height=480, fps=30.0, enforce_aspect="strict")
    res = od.apply_mode_with_verification(cap, req, candidates=[(640, 480)], warmup_grabs=0)

    assert res.accepted is False
    assert (res.width, res.height) == (1280, 720)
