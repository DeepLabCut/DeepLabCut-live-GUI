from __future__ import annotations

import importlib.metadata
import inspect
import os
from pathlib import Path

import numpy as np
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
    params, _ = _get_signature_params(DLCLive.__init__)
    params = {
        name
        for name, p in params.items()
        if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    missing = {name for name in expected if name not in params}
    assert not missing, f"DLCLive.__init__ is missing expected kwargs called by GUI: {sorted(missing)}"


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
    # NOTE: frame_time is passed as a kwarg, so we only check for "frame" as a required param.
    #  This is used by DLCLive Processor classes, rather than the DLCLive class itself.

    init_params, _ = _get_signature_params(DLCLive.init_inference)
    init_missing = {name for name in {"frame"} if name not in init_params}
    assert not init_missing, f"DLCLive.init_inference signature mismatch, missing: {sorted(init_missing)}"

    get_pose_params, _ = _get_signature_params(DLCLive.get_pose)
    get_pose_missing = {name for name in {"frame"} if name not in get_pose_params}
    assert not get_pose_missing, f"DLCLive.get_pose signature mismatch, missing: {sorted(get_pose_missing)}"


@pytest.mark.dlclive_compat
def test_dlclive_minimal_inference_smoke():
    """
    Real runtime smoke test (init + pose call) using a tiny exported model.

    Opt-in via env vars:
    - DLCLIVE_TEST_MODEL_PATH: absolute/relative path to exported model folder/file
    - DLCLIVE_TEST_MODEL_TYPE: optional model type (default: pytorch)
    """
    model_path_env = os.getenv("DLCLIVE_TEST_MODEL_PATH", "").strip()
    if not model_path_env:
        pytest.skip("Set DLCLIVE_TEST_MODEL_PATH to run real DLCLive inference smoke test.")

    model_path = Path(model_path_env).expanduser()
    if not model_path.exists():
        pytest.skip(f"DLCLIVE_TEST_MODEL_PATH does not exist: {model_path}")

    model_type = os.getenv("DLCLIVE_TEST_MODEL_TYPE", "pytorch").strip() or "pytorch"

    from dlclive import DLCLive  # noqa: PLC0415

    from dlclivegui.services.dlc_processor import validate_pose_array  # noqa: PLC0415

    dlc = DLCLive(
        model_path=str(model_path),
        model_type=model_type,
        dynamic=[False, 0.5, 10],
        resize=1.0,
        precision="FP32",
        single_animal=True,
    )

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    dlc.init_inference(frame)
    pose = dlc.get_pose(frame, frame_time=0.0)
    pose_arr = validate_pose_array(pose, source_backend="DLCLive.get_pose")

    assert pose_arr.ndim in (2, 3)
    assert pose_arr.shape[-1] == 3
    assert np.isfinite(pose_arr).all()
