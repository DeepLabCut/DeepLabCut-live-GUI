import numpy as np
import pytest

from dlclivegui.services.dlc_processor import validate_pose_array


@pytest.mark.unit
def test_validate_pose_array_keeps_single_animal_shape():
    pose = np.ones((5, 3), dtype=np.float64)
    out = validate_pose_array(pose)
    assert out.shape == (5, 3)
    assert out.dtype == np.float64


@pytest.mark.unit
def test_validate_pose_array_accepts_multi_animal():
    pose = np.ones((2, 5, 3), dtype=np.float32)
    out = validate_pose_array(pose)
    assert out.shape == (2, 5, 3)


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_pose,expected",
    [
        (np.ones((5, 2), dtype=np.float32), "last dimension size 3"),
        (np.ones((2, 5, 4), dtype=np.float32), "last dimension size 3"),
        (np.ones((3,), dtype=np.float32), "expected a 2D or 3D array"),
    ],
)
def test_validate_pose_array_rejects_invalid_shapes(bad_pose, expected):
    with pytest.raises(ValueError, match=expected):
        validate_pose_array(bad_pose)


@pytest.mark.unit
def test_validate_pose_array_rejects_non_numeric():
    pose = np.array([[["x", "y", "p"]]], dtype=object)
    with pytest.raises(ValueError, match="expected numeric values"):
        validate_pose_array(pose)
