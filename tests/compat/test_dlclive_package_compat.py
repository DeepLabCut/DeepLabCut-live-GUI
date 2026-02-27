from __future__ import annotations

import importlib.metadata
import inspect

import pytest


def _get_signature_params(callable_obj) -> tuple[set[str], bool]:
    """
    Return allowed keyword names for callable, allowing for **kwargs.
    
    Example:
    >>> params, accepts_var_kw = _get_signature_params(lambda x, y, **kwargs: None, {"x", "y"})
    >>> params == {"x", "y"}
    True
    >>> accepts_var_kw
    True
    """
    sig = inspect.signature(callable_obj)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    return params, accepts_var_kw


@pytest.mark.dlclive_compat
def test_dlclive_package_is_importable():
    from dlclive import DLCLive  # noqa: PLC0415

    assert DLCLive is not None
    # Helpful for CI logs to confirm matrix install result.
    _ = importlib.metadata.version("deeplabcut-live")


@pytest.mark.dlclive_compat
def test_dlclive_constructor_accepts_gui_expected_kwargs():
    """
    GUI passes these kwargs when constructing DLCLive.
    This test catches upstream API changes that would break initialization.
    """
    from dlclive import DLCLive  # noqa: PLC0415

    expected = {
        "model_path",
        "model_type",
        "processor",
        "dynamic",
        "resize",
        "precision",
        "single_animal",
        "device",
    }
    params, accepts_var_kw = _get_signature_params(DLCLive.__init__)
    missing = {name for name in expected if name not in params}
    assert not missing, f"DLCLive.__init__ is missing expected kwargs called by GUI: {sorted(missing)}"
    assert accepts_var_kw, "DLCLive.__init__ should accept **kwargs" # captures current behavior


@pytest.mark.dlclive_compat
def test_dlclive_methods_match_gui_usage():
    """
    GUI expects:
    - init_inference(frame)
    - get_pose(frame, frame_time=<float>)
    """
    from dlclive import DLCLive  # noqa: PLC0415

    assert hasattr(DLCLive, "init_inference"), "DLCLive must provide init_inference(frame)"
    assert hasattr(DLCLive, "get_pose"), "DLCLive must provide get_pose(frame, frame_time=...)"

    init_params, _ = _get_signature_params(DLCLive.init_inference)
    init_missing = {name for name in {"frame"} if name not in init_params}
    assert not init_missing, f"DLCLive.init_inference signature mismatch, missing: {sorted(init_missing)}"

    get_pose_params, _ = _get_signature_params(DLCLive.get_pose)
    get_pose_missing = {name for name in {"frame", "frame_time"} if name not in get_pose_params}
    assert not get_pose_missing, f"DLCLive.get_pose signature mismatch, missing: {sorted(get_pose_missing)}"
