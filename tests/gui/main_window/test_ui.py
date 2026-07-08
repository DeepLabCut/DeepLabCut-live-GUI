from __future__ import annotations

import pytest

from dlclivegui.services.multi_camera_controller import get_camera_id


@pytest.mark.gui
class TestCameraLabels:
    def test_update_active_cameras_label_uses_backend_device_name(self, window):
        w = window
        cam = w._config.multi_camera.cameras[0]
        cam.name = ""
        cam.properties = {"fake": {"device_name": "The Camera"}}
        for extra in w._config.multi_camera.cameras[1:]:
            extra.enabled = False

        w._update_active_cameras_label()

        assert "The Camera" in w.active_cameras_label.text()
        assert "[fake:0]" in w.active_cameras_label.text()

    def test_refresh_dlc_camera_list_displays_friendly_label_but_stores_stable_id(self, window):
        w = window
        cam = w._config.multi_camera.cameras[0]
        cam.name = ""
        cam.properties = {"fake": {"device_name": "The Camera", "device_id": "stable-123"}}
        for extra in w._config.multi_camera.cameras[1:]:
            extra.enabled = False

        w._refresh_dlc_camera_list()

        assert w.dlc_camera_combo.count() == 1
        assert "The Camera" in w.dlc_camera_combo.itemText(0)
        assert "stable-123" not in w.dlc_camera_combo.itemText(0)
        assert w.dlc_camera_combo.itemData(0) == get_camera_id(cam)

    def test_label_for_cam_id_prefers_configured_friendly_label(self, window):
        w = window
        cam = w._config.multi_camera.cameras[0]
        cam.name = "Top camera"

        assert w._label_for_cam_id(get_camera_id(cam)).startswith("Top camera")

    def test_label_for_cam_id_uses_runtime_display_id_fallback(self, window):
        w = window
        w._multi_camera_display_ids = {"runtime:id": "Runtime Camera"}

        assert w._label_for_cam_id("runtime:id") == "Runtime Camera"

    def test_label_for_cam_id_unknown_is_neutral(self, window):
        assert window._label_for_cam_id("missing:id") == "Unknown camera"
