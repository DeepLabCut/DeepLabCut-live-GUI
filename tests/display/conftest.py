import numpy as np
import pytest


@pytest.fixture
def test_frame(h, w, c=3, value=0, dtype=np.uint8):
    """Helper to create test frames with predictable content."""
    if c == 1:
        return (np.ones((h, w), dtype=dtype) * value).astype(dtype)
    return (np.ones((h, w, c), dtype=dtype) * value).astype(dtype)
