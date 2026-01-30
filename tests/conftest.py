# tests/dlc/conftest.py

from __future__ import annotations

import time

import numpy as np
import pytest

from dlclivegui.cameras import CameraFactory
from dlclivegui.cameras.base import CameraBackend
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


class FakeBackend(CameraBackend):
    def __init__(self, settings):
        super().__init__(settings)
        self._opened = False
        self._counter = 0

    @classmethod
    def is_available(cls) -> bool:
        return True

    def open(self) -> None:
        self._opened = True

    def read(self):
        # Produce a deterministic small frame
        if not self._opened:
            raise RuntimeError("not opened")
        self._counter += 1
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        ts = time.time()
        return frame, ts

    def close(self) -> None:
        self._opened = False

    def stop(self) -> None:
        pass


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def patch_factory(monkeypatch):
    def _create(settings):
        return FakeBackend(settings)

    monkeypatch.setattr(CameraFactory, "create", staticmethod(_create))
    return _create


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
