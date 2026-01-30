# tests/dlc/conftest.py

from __future__ import annotations

import numpy as np
import pytest

from dlclivegui.config import DLCProcessorSettings
from dlclivegui.utils.config_models import DLCProcessorSettingsModel

# ---------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------


class FakeDLCLive:
    """A minimal fake DLCLive object for testing."""

    def __init__(self, **opts):
        self.opts = opts
        self.init_called = False
        self.pose_calls = 0

    def init_inference(self, frame):
        self.init_called = True

    def get_pose(self, frame, frame_time=None):
        self.pose_calls += 1
        # Deterministic small pose array
        return np.ones((2, 2), dtype=float)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def monkeypatch_dlclive(monkeypatch):
    """
    Replace the dlclive.DLCLive import with FakeDLCLive *within* the dlc_processor module.

    Scope is function-level by default, which keeps tests isolated.
    """
    from dlclivegui.services import dlc_processor

    monkeypatch.setattr(dlc_processor, "DLCLive", FakeDLCLive)
    return FakeDLCLive


@pytest.fixture
def settings_dc():
    """A standard DLCProcessorSettings dataclass for tests."""
    return DLCProcessorSettings(model_path="dummy.pt")


@pytest.fixture
def settings_model():
    """A standard Pydantic DLCProcessorSettingsModel for tests."""
    return DLCProcessorSettingsModel(model_path="dummy.pt")
