# tests/display/conftest.py
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest


@pytest.fixture
def test_frame() -> Callable[..., np.ndarray]:
    """Return a factory for creating predictable test frames."""

    def make_frame(
        h: int,
        w: int,
        c: int = 3,
        value: int | float = 0,
        dtype: np.dtype = np.uint8,
    ) -> np.ndarray:
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        shape = (h, w) if c == 1 else (h, w, c)
        return np.full(shape, value, dtype=dtype)

    return make_frame
