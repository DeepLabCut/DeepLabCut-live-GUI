"""Camera configuration dialog for multi-camera setup (with async preview loading)."""

# dlclivegui/gui/camera_config/camera_config_dialog.py
from __future__ import annotations

import copy
import logging

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QListWidgetItem,
    QMessageBox,
    QScrollArea,
    QStyle,
    QWidget,
)

from ...cameras.factory import CameraFactory, DetectedCamera, apply_detected_identity, camera_identity_key
from ...config import CameraSettings, MultiCameraSettings
from .loaders import CameraLoadWorker, CameraProbeWorker, CameraScanState, DetectCamerasWorker
from .preview import PreviewSession, PreviewState, apply_crop, apply_rotation, resize_to_fit, to_display_pixmap
from .ui_blocks import setup_camera_config_dialog_ui

LOGGER = logging.getLogger(__name__)


class CameraConfigDialog(QDialog):
    """Dialog for configuring multiple cameras with async preview loading."""

    MAX_CAMERAS = 4
    settings_changed = Signal(object)  # MultiCameraSettingsModel
    # Camera discovery signals
    scan_started = Signal(str)
    scan_finished = Signal()

    # -------------------------------
    # Constructor, properties, Qt lifecycle
    # -------------------------------

    def __init__(
        self,
        parent: QWidget | None = None,
        multi_camera_settings: MultiCameraSettings | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Configure Cameras")
        self.setMinimumSize(960, 720)

        self._dlc_camera_id: str | None = None
        # self.dlc_camera_id: str | None = None
        # Actual/working camera settings
        self._multi_camera_settings = multi_camera_settings or MultiCameraSettings(cameras=[])
        self._working_settings = self._multi_camera_settings.model_copy(deep=True)
        self._detected_cameras: list[DetectedCamera] = []
        self._probe_apply_to_requested: bool = False
        self._probe_target_row: int | None = None
        self._current_edit_index: int | None = None
        self._suppress_selection_actions: bool = False

        # Preview state
        self._preview: PreviewSession = PreviewSession()

        # Camera detection worker
        self._scan_worker: DetectCamerasWorker | None = None
        self._scan_state: CameraScanState = CameraScanState.IDLE

        # UI elements for eventFilter (assigned in _setup_ui)
        self._settings_scroll: QScrollArea | None = None
        self._settings_scroll_contents: QWidget | None = None

        self._setup_ui()
        self._populate_from_settings()
        self._connect_signals()

    @property
    def dlc_camera_id(self) -> str | None:
        """Get the currently selected DLC camera ID."""
        return self._dlc_camera_id

    @dlc_camera_id.setter
    def dlc_camera_id(self, value: str | None) -> None:
        """Set the currently selected DLC camera ID."""
        self._dlc_camera_id = value
        self._refresh_camera_labels()

    def showEvent(self, event):
        super().showEvent(event)
        # Reset cleanup guard so close cleanup runs for each session
        self._cleanup_done = False

        # Rebuild the working copy from the latest “accepted” settings
        self._working_settings = self._multi_camera_settings.model_copy(deep=True)
        self._current_edit_index = None

    # Maintain overlay geometry when resizing
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_loading_overlay") and self._loading_overlay.isVisible():
            self._position_loading_overlay()

    def eventFilter(self, obj, event):
        # --- Keep scroll contents locked to viewport width (prevents horizontal scrolling/clipping) ---
        if (
            hasattr(self, "_settings_scroll")
            and self._settings_scroll is not None
            and obj is self._settings_scroll.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            try:
                if self._settings_scroll_contents is not None:
                    vw = self._settings_scroll.viewport().width()
                    # Set minimum width to viewport width to force wrapping/reflow instead of horizontal overflow
                    self._settings_scroll_contents.setMinimumWidth(vw)
            except Exception:
                pass
            return False  # allow normal processing

        # Keep your existing overlay resize handling
        if obj is self.available_cameras_list and event.type() == QEvent.Type.Resize:
            if self._scan_overlay and self._scan_overlay.isVisible():
                self._position_scan_overlay()
            return super().eventFilter(obj, event)

        # Intercept Enter in FPS and crop spinboxes
        if event.type() == QEvent.KeyPress and isinstance(event, QKeyEvent):
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if obj in (
                    self.cam_fps,
                    self.cam_width,
                    self.cam_height,
                    self.cam_exposure,
                    self.cam_gain,
                    self.cam_crop_x0,
                    self.cam_crop_y0,
                    self.cam_crop_x1,
                    self.cam_crop_y1,
                ):
                    # Commit any pending text → value
                    try:
                        obj.interpretText()
                    except Exception:
                        pass
                    # Apply settings to persist crop/FPS to CameraSettings
                    self._apply_camera_settings()
                    # Consume so OK isn't triggered
                    return True

        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        """Handle dialog close event to ensure cleanup."""
        self._on_close_cleanup()
        super().closeEvent(event)

    def reject(self) -> None:
        """Handle dialog rejection (Cancel or close)."""
        self._on_close_cleanup()
        super().reject()

    def _on_close_cleanup(self) -> None:
        """Stop preview, cancel workers, and reset scan UI. Safe to call multiple times."""
        # Guard to avoid running twice if closeEvent + reject/accept both run
        if getattr(self, "_cleanup_done", False):
            return
        self._cleanup_done = True

        # Stop preview (loader + backend + timer)
        try:
            self._stop_preview()
        except Exception:
            LOGGER.exception("Cleanup: failed stopping preview")

        # Cancel scan worker
        sw = getattr(self, "_scan_worker", None)
        if sw and sw.isRunning():
            try:
                sw.requestInterruption()
            except Exception:
                pass
            # Keep this short to reduce UI freeze
            sw.wait(300)
        self._set_scan_state(CameraScanState.IDLE)
        if self._scan_worker and not self._scan_worker.isRunning():
            self._cleanup_scan_worker()

        # Cancel probe worker
        pw = getattr(self, "_probe_worker", None)
        if pw and pw.isRunning():
            try:
                pw.request_cancel()
            except Exception:
                pass
            pw.wait(300)
        self._probe_worker = None

        # Hide overlays / reset UI bits
        try:
            self._hide_scan_overlay()
        except Exception:
            pass

        # Defensive: some widgets may not exist depending on UI setup timing
        for w, visible, enabled in (
            ("scan_progress", False, None),
            ("scan_cancel_btn", False, True),
            ("refresh_btn", None, True),
            ("backend_combo", None, True),
        ):
            widget = getattr(self, w, None)
            if widget is None:
                continue
            if visible is not None:
                widget.setVisible(visible)
            if enabled is not None:
                widget.setEnabled(enabled)

        try:
            self._sync_scan_ui()
        except Exception:
            pass

    # -------------------------------
    # UI setup
    # -------------------------------
    def _setup_ui(self) -> None:
        setup_camera_config_dialog_ui(self)

    def _position_scan_overlay(self) -> None:
        """Position scan overlay to cover the available_cameras_list area."""
        if not self._scan_overlay or not self.available_cameras_list:
            return
        parent = self._scan_overlay.parent()  # available_group
        top_left = self.available_cameras_list.mapTo(parent, self.available_cameras_list.rect().topLeft())
        rect = self.available_cameras_list.rect()
        self._scan_overlay.setGeometry(top_left.x(), top_left.y(), rect.width(), rect.height())

    def _show_scan_overlay(self, message: str = "Discovering cameras…") -> None:
        self._scan_overlay.setText(message)
        self._scan_overlay.setVisible(True)
        self._position_scan_overlay()

    def _hide_scan_overlay(self) -> None:
        self._scan_overlay.setVisible(False)

    def _position_loading_overlay(self):
        # Cover just the preview image area (label), not the whole group
        if not self.preview_label:
            return
        gp = self.preview_label.mapTo(self.preview_group, self.preview_label.rect().topLeft())
        rect = self.preview_label.rect()
        self._loading_overlay.setGeometry(gp.x(), gp.y(), rect.width(), rect.height())

    # -------------------------------
    # Signal setup
    # -------------------------------
    def _connect_signals(self) -> None:
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self.refresh_btn.clicked.connect(self._refresh_available_cameras)
        self.add_camera_btn.clicked.connect(self._add_selected_camera)
        self.remove_camera_btn.clicked.connect(self._remove_selected_camera)
        self.move_up_btn.clicked.connect(self._move_camera_up)
        self.move_down_btn.clicked.connect(self._move_camera_down)
        self.active_cameras_list.currentRowChanged.connect(self._on_active_camera_selected)
        self.available_cameras_list.currentRowChanged.connect(self._on_available_camera_selected)
        self.available_cameras_list.itemDoubleClicked.connect(self._on_available_camera_double_clicked)
        self.apply_settings_btn.clicked.connect(self._apply_camera_settings)
        self.reset_settings_btn.clicked.connect(self._reset_selected_camera)
        self.preview_btn.clicked.connect(self._toggle_preview)
        self.ok_btn.clicked.connect(self._on_ok_clicked)
        self.cancel_btn.clicked.connect(self.reject)
        self.scan_started.connect(lambda _: setattr(self, "_dialog_active", True))
        self.scan_finished.connect(lambda: setattr(self, "_dialog_active", False))
        self.scan_cancel_btn.clicked.connect(self.request_scan_cancel)

        def _mark_dirty(*_args):
            self.apply_settings_btn.setEnabled(True)
            self._set_apply_dirty(True)

        for sb in (
            self.cam_fps,
            self.cam_crop_x0,
            self.cam_crop_y0,
            self.cam_crop_x1,
            self.cam_crop_y1,
            self.cam_exposure,
            self.cam_gain,
            self.cam_width,
            self.cam_height,
        ):
            if hasattr(sb, "valueChanged"):
                sb.valueChanged.connect(_mark_dirty)

        self.cam_rotation.currentIndexChanged.connect(lambda *_: _mark_dirty())
        self.cam_enabled_checkbox.stateChanged.connect(lambda *_: _mark_dirty())

    # -------------------------------
    # UI state updates
    # -------------------------------
    def _is_preview_live(self) -> bool:
        return self._preview.state in (PreviewState.ACTIVE, PreviewState.LOADING)

    def _set_apply_dirty(self, dirty: bool) -> None:
        """Visually mark Apply Settings button as 'dirty' (pending edits)."""
        if dirty:
            self.apply_settings_btn.setText("Apply Settings *")
            self.apply_settings_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning))
            self.apply_settings_btn.setToolTip("You have unapplied changes. Click to apply them.")
        else:
            self.apply_settings_btn.setText("Apply Settings")
            self.apply_settings_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
            self.apply_settings_btn.setToolTip("")

    def _update_button_states(self) -> None:
        scan_running = self._is_scan_running()

        active_row = self.active_cameras_list.currentRow()
        has_active_selection = active_row >= 0

        # Allow removing/moving active cameras even during scanning
        self.remove_camera_btn.setEnabled(has_active_selection)
        self.move_up_btn.setEnabled(has_active_selection and active_row > 0)
        self.move_down_btn.setEnabled(has_active_selection and active_row < self.active_cameras_list.count() - 1)

        self.preview_btn.setEnabled(has_active_selection or self._preview.state == PreviewState.LOADING)

        available_row = self.available_cameras_list.currentRow()
        self.add_camera_btn.setEnabled(available_row >= 0 and not scan_running)

    def _sync_preview_ui(self) -> None:
        """Update buttons/overlays based on preview state only."""
        st = self._preview.state

        if st == PreviewState.LOADING:
            self._set_preview_button_loading(True)
            self.preview_btn.setEnabled(True)
            self.preview_group.setVisible(True)
        elif st == PreviewState.ACTIVE:
            self._set_preview_button_loading(False)
            self.preview_btn.setText("Stop Preview")
            self.preview_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
            self.preview_btn.setEnabled(True)
            self.preview_group.setVisible(True)
        else:  # IDLE / STOPPING / ERROR
            self._set_preview_button_loading(False)
            self.preview_btn.setText("Start Preview")
            self.preview_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.preview_btn.setEnabled(self.active_cameras_list.currentRow() >= 0)
            self.preview_group.setVisible(False)

        self._update_button_states()

    def _set_detected_labels(self, cam: CameraSettings) -> None:
        """Update the read-only detected labels based on cam.properties[backend]."""
        backend = (cam.backend or "").lower()
        props = cam.properties if isinstance(cam.properties, dict) else {}
        ns = props.get(backend, {}) if isinstance(props.get(backend, None), dict) else {}

        det_res = ns.get("detected_resolution")
        det_fps = ns.get("detected_fps")

        if isinstance(det_res, (list, tuple)) and len(det_res) == 2:
            try:
                w, h = int(det_res[0]), int(det_res[1])
                self.detected_resolution_label.setText(f"{w}×{h}")
            except Exception:
                self.detected_resolution_label.setText("—")
        else:
            self.detected_resolution_label.setText("—")

        if isinstance(det_fps, (int, float)) and float(det_fps) > 0:
            self.detected_fps_label.setText(f"{float(det_fps):.2f}")
        else:
            self.detected_fps_label.setText("—")

    def _refresh_camera_labels(self) -> None:
        cam_list = getattr(self, "active_cameras_list", None)
        if not cam_list:
            return

        cam_list.blockSignals(True)  # prevent unwanted selection change events during update
        try:
            for i in range(cam_list.count()):
                item = cam_list.item(i)
                cam = item.data(Qt.ItemDataRole.UserRole)
                if cam:
                    item.setText(self._format_camera_label(cam, i))
        finally:
            cam_list.blockSignals(False)

    def _format_camera_label(self, cam: CameraSettings, index: int = -1) -> str:
        status = "✓" if cam.enabled else "○"
        this_id = f"{(cam.backend or '').lower()}:{cam.index}"
        dlc_indicator = " [DLC]" if this_id == self._dlc_camera_id and cam.enabled else ""
        return f"{status} {cam.name} [{cam.backend}:{cam.index}]{dlc_indicator}"

    def _update_active_list_item(self, row: int, cam: CameraSettings) -> None:
        """Refresh the active camera list row text and color."""
        item = self.active_cameras_list.item(row)
        if not item:
            return
        self._suppress_selection_actions = True  # prevent unwanted selection change events during update
        try:
            item.setText(self._format_camera_label(cam, row))
            item.setData(Qt.ItemDataRole.UserRole, cam)
            item.setForeground(Qt.GlobalColor.gray if not cam.enabled else Qt.GlobalColor.black)
            self._refresh_camera_labels()
            self._update_button_states()
        finally:
            self._suppress_selection_actions = False

    def _update_controls_for_backend(self, backend_name: str) -> None:
        backend_key = (backend_name or "opencv").lower()
        caps = CameraFactory.backend_capabilities(backend_key)

        def apply(widget, feature: str, label: str, *, allow_best_effort: bool = True):
            level = caps.get(feature, None)
            if level is None:
                widget.setEnabled(False)
                widget.setToolTip(f"{label} is not supported by this backend.")
                return

            if level.value == "unsupported":
                widget.setEnabled(False)
                widget.setToolTip(f"{label} is not supported by the {backend_key} backend.")
            elif level.value == "best_effort":
                widget.setEnabled(bool(allow_best_effort))
                widget.setToolTip(f"{label} is best-effort in {backend_key}. Some cameras/drivers may ignore it.")
            else:  # supported
                widget.setEnabled(True)
                widget.setToolTip("")

        # Resolution controls
        apply(self.cam_width, "set_resolution", "Resolution")
        apply(self.cam_height, "set_resolution", "Resolution")

        # FPS
        apply(self.cam_fps, "set_fps", "Frame rate")

        # Exposure / Gain
        apply(self.cam_exposure, "set_exposure", "Exposure")
        apply(self.cam_gain, "set_gain", "Gain")

    def _set_preview_button_loading(self, loading: bool) -> None:
        if loading:
            self.preview_btn.setText("Cancel Loading")
            self.preview_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserStop))
        else:
            self.preview_btn.setText("Start Preview")
            self.preview_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def _show_loading_overlay(self, message: str) -> None:
        self._loading_overlay.setText(message)
        self._loading_overlay.setVisible(True)
        self._position_loading_overlay()

    def _hide_loading_overlay(self) -> None:
        self._loading_overlay.setVisible(False)

    def _append_status(self, text: str) -> None:
        LOGGER.debug(f"Preview status: {text}")
        self.preview_status.append(text)
        self.preview_status.moveCursor(QTextCursor.End)
        self.preview_status.ensureCursorVisible()

    # -------------------------------
    # Camera discovery and probing
    # -------------------------------
    def _on_backend_changed(self, _index: int) -> None:
        self._refresh_available_cameras()

    def _is_scan_running(self) -> bool:
        if self._scan_state in (CameraScanState.RUNNING, CameraScanState.CANCELING):
            return True
        w = self._scan_worker
        return bool(w and w.isRunning())

    def _set_scan_state(self, state: CameraScanState, message: str | None = None) -> None:
        """Single source of truth for scan-related UI controls."""
        self._scan_state = state

        scanning = state in (CameraScanState.RUNNING, CameraScanState.CANCELING)

        # Overlay message
        if scanning:
            self._show_scan_overlay(
                message or ("Canceling discovery…" if state == CameraScanState.CANCELING else "Discovering cameras…")
            )
        else:
            self._hide_scan_overlay()

        # Progress + cancel controls
        self.scan_progress.setVisible(scanning)
        if scanning:
            self.scan_progress.setRange(0, 0)  # indeterminate
        self.scan_cancel_btn.setVisible(scanning)
        self.scan_cancel_btn.setEnabled(state == CameraScanState.RUNNING)  # disabled while canceling

        # Disable discovery inputs while scanning
        self.backend_combo.setEnabled(not scanning)
        self.refresh_btn.setEnabled(not scanning)

        # Available list + add flow blocked while scanning (structure edits disallowed)
        self.available_cameras_list.setEnabled(not scanning)
        self.add_camera_btn.setEnabled(False if scanning else (self.available_cameras_list.currentRow() >= 0))

        self._update_button_states()

    def _cleanup_scan_worker(self) -> None:
        # worker is truly finished now
        w = self._scan_worker
        self._scan_worker = None
        if w is not None:
            w.deleteLater()

    def _finish_scan(self, reason: str) -> None:
        """Mark scan UX complete (idempotent) and emit scan_finished queued."""
        if self._scan_state in (CameraScanState.DONE, CameraScanState.IDLE):
            return

        # Transition scan UX to DONE (UI controls restored)
        self._set_scan_state(CameraScanState.DONE)

        QTimer.singleShot(0, self.scan_finished.emit)

        LOGGER.debug("[Scan] finished reason=%s", reason)

    def _refresh_available_cameras(self) -> None:
        """Refresh the list of available cameras asynchronously."""
        backend = self.backend_combo.currentData() or self.backend_combo.currentText().split()[0]

        if self._is_scan_running():
            self._show_scan_overlay("Already discovering cameras…")
            return

        # Reset UI/list
        self.available_cameras_list.clear()
        self._detected_cameras = []

        self._set_scan_state(CameraScanState.RUNNING, message=f"Discovering {backend} cameras…")

        # Start worker
        w = DetectCamerasWorker(backend, max_devices=10, parent=self)
        self._scan_worker = w

        w.progress.connect(self._on_scan_progress)
        w.result.connect(self._on_scan_result)
        w.error.connect(self._on_scan_error)
        w.canceled.connect(self._on_scan_canceled)

        # Cleanup only
        w.finished.connect(self._cleanup_scan_worker)

        self.scan_started.emit(f"Scanning {backend} cameras…")
        w.start()

    def _on_scan_progress(self, msg: str) -> None:
        if self.sender() is not self._scan_worker:
            LOGGER.debug("[Scan] Ignoring progress from old worker: %s", msg)
            return
        if self._scan_state not in (CameraScanState.RUNNING, CameraScanState.CANCELING):
            return
        self._show_scan_overlay(msg or "Discovering cameras…")

    def _on_scan_result(self, cams: list) -> None:
        if self.sender() is not self._scan_worker:
            LOGGER.debug("[Scan] Ignoring result from old worker: %d cameras", len(cams) if cams else 0)
            return
        if self._scan_state not in (CameraScanState.RUNNING, CameraScanState.CANCELING):
            return

        # Apply results to UI first (stability guarantee)
        self._detected_cameras = cams or []
        self.available_cameras_list.clear()

        if not self._detected_cameras:
            placeholder = QListWidgetItem("No cameras detected.")
            placeholder.setFlags(Qt.ItemIsEnabled)
            self.available_cameras_list.addItem(placeholder)
        else:
            for cam in self._detected_cameras:
                item = QListWidgetItem(f"{cam.label} (index {cam.index})")
                item.setData(Qt.ItemDataRole.UserRole, cam)
                self.available_cameras_list.addItem(item)
            self.available_cameras_list.setCurrentRow(0)

        # Now UI is stable: finish scan UX and emit scan_finished queued
        self._finish_scan("result")

    def _on_scan_error(self, msg: str) -> None:
        if self.sender() is not self._scan_worker:
            LOGGER.debug("[Scan] Ignoring error from old worker: %s", msg)
            return
        if self._scan_state not in (CameraScanState.RUNNING, CameraScanState.CANCELING):
            return

        QMessageBox.warning(self, "Camera Scan", f"Failed to detect cameras:\n{msg}")

        # Ensure UI is stable (list is stable even if empty) before finishing
        if self.available_cameras_list.count() == 0:
            placeholder = QListWidgetItem("Scan failed.")
            placeholder.setFlags(Qt.ItemIsEnabled)
            self.available_cameras_list.addItem(placeholder)

        self._finish_scan("error")

    def request_scan_cancel(self) -> None:
        if not self._is_scan_running():
            return

        self._set_scan_state(CameraScanState.CANCELING, message="Canceling discovery…")

        w = self._scan_worker
        if w is not None:
            try:
                w.requestInterruption()
            except Exception:
                pass

        # Guarantee UI stability before scan_finished:
        if self.available_cameras_list.count() == 0:
            placeholder = QListWidgetItem("Scan canceled.")
            placeholder.setFlags(Qt.ItemIsEnabled)
            self.available_cameras_list.addItem(placeholder)

        if w is None or not w.isRunning():
            self._finish_scan("cancel")

    def _on_scan_canceled(self) -> None:
        if self.sender() is not self._scan_worker:
            LOGGER.debug("[Scan] Ignoring canceled signal from old worker.")
            return
        self._set_scan_state(CameraScanState.CANCELING, message="Finalizing cancellation…")
        # If cancel is requested without clicking cancel (e.g., dialog closing), ensure UI finishes
        if self._scan_state in (CameraScanState.RUNNING, CameraScanState.CANCELING):
            if self.available_cameras_list.count() == 0:
                placeholder = QListWidgetItem("Scan canceled.")
                placeholder.setFlags(Qt.ItemIsEnabled)
                self.available_cameras_list.addItem(placeholder)
            self._finish_scan("canceled")

    def _on_available_camera_selected(self, row: int) -> None:
        if self._scan_worker and self._scan_worker.isRunning():
            self.add_camera_btn.setEnabled(False)
            return
        item = self.available_cameras_list.item(row) if row >= 0 else None
        detected = item.data(Qt.ItemDataRole.UserRole) if item else None
        self.add_camera_btn.setEnabled(isinstance(detected, DetectedCamera))

    def _on_available_camera_double_clicked(self, item: QListWidgetItem) -> None:
        if self._is_scan_running():
            return
        self._add_selected_camera()

    # -------------------------------
    # Active camera selection and list
    # -------------------------------
    def _on_active_camera_selected(self, row: int) -> None:
        if getattr(self, "_suppress_selection_actions", False):
            LOGGER.debug("[Selection] Suppressed currentRowChanged event at index %d.", row)
            return
        prev_row = self._current_edit_index
        LOGGER.debug(
            "[Select] row=%s prev=%s preview_state=%s",
            row,
            prev_row,
            self._preview.state,
        )
        if row is None or row < 0:
            LOGGER.debug(
                "[Selection] row<0 (selection cleared) ignored to avoid"
                " stopping preview/loading when clicking away. row=%s",
                row,
            )
            return

        # If row is the same, ignore
        if prev_row is not None and prev_row == row:
            LOGGER.debug("[Selection] Redundant currentRowChanged to same index %d; ignoring.", row)
            return

        # If switching away from a previous camera, commit pending edits first
        if prev_row is not None and prev_row != row:
            if not self._commit_pending_edits(reason="before switching camera selection"):
                # Revert selection back to previous row so the user stays on the invalid camera
                try:
                    self.active_cameras_list.blockSignals(True)
                    self.active_cameras_list.setCurrentRow(prev_row)
                finally:
                    self.active_cameras_list.blockSignals(False)
                return

        # Stop any running preview when selection changes
        if self._is_preview_live():
            self._stop_preview()

        self._current_edit_index = row

        self._update_button_states()
        if row < 0 or row >= self.active_cameras_list.count():
            self._clear_settings_form()
            return
        item = self.active_cameras_list.item(row)
        cam = item.data(Qt.ItemDataRole.UserRole)
        if cam:
            self.apply_settings_btn.setEnabled(True)
            self.reset_settings_btn.setEnabled(True)
            self._load_camera_to_form(cam)
            self._start_probe_for_camera(cam, apply_to_requested=False)

    def _add_selected_camera(self) -> None:
        if not self._commit_pending_edits(reason="before adding a new camera"):
            return
        row = self.available_cameras_list.currentRow()
        if row < 0:
            return
        # limit check
        active_count = len(
            [
                i
                for i in range(self.active_cameras_list.count())
                if self.active_cameras_list.item(i).data(Qt.ItemDataRole.UserRole).enabled
            ]
        )
        if active_count >= self.MAX_CAMERAS:
            QMessageBox.warning(self, "Maximum Cameras", f"Maximum of {self.MAX_CAMERAS} active cameras allowed.")
            return
        item = self.available_cameras_list.item(row)
        detected = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(detected, DetectedCamera):
            QMessageBox.warning(self, "Invalid Selection", "Selected item is not a valid camera.")
            return
        # make sure this is to lower for comparison against camera_identity_key
        backend = (self.backend_combo.currentData() or "opencv").lower()

        det_key = None
        if getattr(detected, "device_id", None):
            det_key = (backend, "device_id", detected.device_id)
        else:
            det_key = (backend, "index", int(detected.index))

        for i in range(self.active_cameras_list.count()):
            existing_cam = self.active_cameras_list.item(i).data(Qt.ItemDataRole.UserRole)
            if camera_identity_key(existing_cam) == det_key:
                QMessageBox.warning(self, "Duplicate Camera", "This camera is already in the active list.")
                return

        new_cam = CameraSettings(
            name=detected.label,
            index=detected.index,
            width=0,
            height=0,
            fps=0.0,
            backend=backend,
            exposure=0,
            gain=0.0,
            enabled=True,
            properties={},
        )
        apply_detected_identity(new_cam, detected, backend)
        self._working_settings.cameras.append(new_cam)
        new_index = len(self._working_settings.cameras) - 1
        new_item = QListWidgetItem(self._format_camera_label(new_cam, new_index))
        new_item.setData(Qt.ItemDataRole.UserRole, new_cam)
        self.active_cameras_list.addItem(new_item)
        self.active_cameras_list.setCurrentItem(new_item)
        self._refresh_camera_labels()
        self._update_button_states()
        self._start_probe_for_camera(new_cam)

    def _remove_selected_camera(self) -> None:
        if self._is_preview_live():
            self._stop_preview()
        if not self._commit_pending_edits(reason="before removing a camera"):
            return
        row = self.active_cameras_list.currentRow()
        if row < 0:
            return
        self.active_cameras_list.takeItem(row)
        if row < len(self._working_settings.cameras):
            del self._working_settings.cameras[row]
        self._current_edit_index = None
        self._clear_settings_form()
        self._refresh_camera_labels()
        self._update_button_states()

    def _move_camera_up(self) -> None:
        if self._is_preview_live():
            self._stop_preview()
        if not self._commit_pending_edits(reason="before reordering cameras"):
            return
        row = self.active_cameras_list.currentRow()
        if row <= 0:
            return
        item = self.active_cameras_list.takeItem(row)
        self.active_cameras_list.insertItem(row - 1, item)
        self.active_cameras_list.setCurrentRow(row - 1)
        cams = self._working_settings.cameras
        cams[row], cams[row - 1] = cams[row - 1], cams[row]
        self._refresh_camera_labels()

    def _move_camera_down(self) -> None:
        if self._is_preview_live():
            self._stop_preview()
        if not self._commit_pending_edits(reason="before reordering cameras"):
            return
        row = self.active_cameras_list.currentRow()
        if row < 0 or row >= self.active_cameras_list.count() - 1:
            return
        item = self.active_cameras_list.takeItem(row)
        self.active_cameras_list.insertItem(row + 1, item)
        self.active_cameras_list.setCurrentRow(row + 1)
        cams = self._working_settings.cameras
        cams[row], cams[row + 1] = cams[row + 1], cams[row]
        self._refresh_camera_labels()

    # -------------------------------
    # Form/model mapping & settings application
    # -------------------------------
    def _build_model_from_form(self, base: CameraSettings) -> CameraSettings:
        # construct a dict from form widgets; Pydantic will coerce/validate
        payload = base.model_dump()
        payload.update(
            {
                "enabled": bool(self.cam_enabled_checkbox.isChecked()),
                "width": int(self.cam_width.value()),
                "height": int(self.cam_height.value()),
                "fps": float(self.cam_fps.value()),
                "exposure": int(self.cam_exposure.value()) if self.cam_exposure.isEnabled() else 0,
                "gain": float(self.cam_gain.value()) if self.cam_gain.isEnabled() else 0.0,
                "rotation": int(self.cam_rotation.currentData() or 0),
                "crop_x0": int(self.cam_crop_x0.value()),
                "crop_y0": int(self.cam_crop_y0.value()),
                "crop_x1": int(self.cam_crop_x1.value()),
                "crop_y1": int(self.cam_crop_y1.value()),
            }
        )
        #  Validate and coerce; if invalid, Pydantic will raise
        return CameraSettings.model_validate(payload)

    def _load_camera_to_form(self, cam: CameraSettings) -> None:
        block = [
            self.cam_enabled_checkbox,
            self.cam_width,
            self.cam_height,
            self.cam_fps,
            self.cam_exposure,
            self.cam_gain,
            self.cam_rotation,
            self.cam_crop_x0,
            self.cam_crop_y0,
            self.cam_crop_x1,
            self.cam_crop_y1,
        ]
        for widget in block:
            if hasattr(widget, "blockSignals"):
                widget.blockSignals(True)
        try:
            backend = (cam.backend or "").lower()
            props = cam.properties if isinstance(cam.properties, dict) else {}
            ns = props.get(backend, {}) if isinstance(props, dict) else {}
            self.cam_enabled_checkbox.setChecked(cam.enabled)
            self.cam_name_label.setText(cam.name)
            self.cam_device_name_label.setText(str(ns.get("device_id", "")))
            self.cam_index_label.setText(str(cam.index))
            self.cam_backend_label.setText(cam.backend)
            self._update_controls_for_backend(cam.backend)
            self.cam_width.setValue(cam.width)
            self.cam_height.setValue(cam.height)
            self.cam_fps.setValue(cam.fps)
            self.cam_exposure.setValue(cam.exposure)
            self.cam_gain.setValue(cam.gain)
            rot_index = self.cam_rotation.findData(cam.rotation)
            if rot_index >= 0:
                self.cam_rotation.setCurrentIndex(rot_index)
            self.cam_crop_x0.setValue(cam.crop_x0)
            self.cam_crop_y0.setValue(cam.crop_y0)
            self.cam_crop_x1.setValue(cam.crop_x1)
            self.cam_crop_y1.setValue(cam.crop_y1)
            self.apply_settings_btn.setEnabled(True)
            self._set_detected_labels(cam)
        finally:
            for widget in block:
                if hasattr(widget, "blockSignals"):
                    widget.blockSignals(False)

        self.apply_settings_btn.setEnabled(False)
        self._set_apply_dirty(False)

    def _write_form_to_cam(self, cam: CameraSettings) -> None:
        cam.enabled = bool(self.cam_enabled_checkbox.isChecked())
        cam.width = int(self.cam_width.value())
        cam.height = int(self.cam_height.value())
        cam.fps = float(self.cam_fps.value())
        cam.exposure = int(self.cam_exposure.value() if self.cam_exposure.isEnabled() else 0)
        cam.gain = float(self.cam_gain.value() if self.cam_gain.isEnabled() else 0.0)
        cam.rotation = int(self.cam_rotation.currentData() or 0)
        cam.crop_x0 = int(self.cam_crop_x0.value())
        cam.crop_y0 = int(self.cam_crop_y0.value())
        cam.crop_x1 = int(self.cam_crop_x1.value())
        cam.crop_y1 = int(self.cam_crop_y1.value())

    def _commit_pending_edits(self, *, reason: str = "") -> bool:
        """
        Auto-apply pending edits (if any) before context-changing actions.
        Returns True if it's safe to proceed, False if validation failed.
        """
        # No selection → nothing to commit
        if self._current_edit_index is None or self._current_edit_index < 0:
            return True

        # If Apply button isn't enabled, assume no pending edits
        if not self.apply_settings_btn.isEnabled():
            return True

        try:
            self._append_status(f"[Auto-Apply] Committing pending edits ({reason})…")
            ok = self._apply_camera_settings()
            return bool(ok)
        except Exception as exc:
            # _apply_camera_settings already shows a QMessageBox in many cases,
            # but we add a clear guardrail here in case it doesn't.
            QMessageBox.warning(
                self,
                "Unsaved / Invalid Settings",
                "Your current camera settings are not valid and cannot be applied yet.\n\n"
                "Please fix the highlighted fields (e.g. crop rectangle) or press Reset.\n\n"
                f"Details: {exc}",
            )
            return False

    def _enabled_count_with(self, row: int, new_enabled: bool) -> int:
        count = 0
        for i, cam in enumerate(self._working_settings.cameras):
            enabled = new_enabled if i == row else bool(cam.enabled)
            if enabled:
                count += 1
        return count

    def _apply_camera_settings(self) -> bool:
        try:
            for sb in (
                self.cam_fps,
                self.cam_crop_x0,
                self.cam_width,
                self.cam_height,
                self.cam_exposure,
                self.cam_gain,
                self.cam_crop_y0,
                self.cam_crop_x1,
                self.cam_crop_y1,
            ):
                try:
                    if hasattr(sb, "interpretText"):
                        sb.interpretText()
                except Exception:
                    pass
            if self._current_edit_index is None:
                return True
            row = self._current_edit_index
            if row < 0 or row >= len(self._working_settings.cameras):
                return True

            current_model = self._working_settings.cameras[row]
            new_model = self._build_model_from_form(current_model)

            if bool(new_model.enabled):
                if self._enabled_count_with(row, True) > self.MAX_CAMERAS:
                    QMessageBox.warning(
                        self, "Maximum Cameras", f"Maximum of {self.MAX_CAMERAS} active cameras allowed."
                    )
                    self.cam_enabled_checkbox.setChecked(bool(current_model.enabled))
                    return False

            diff = CameraSettings.check_diff(current_model, new_model)

            LOGGER.debug(
                "[Apply] backend=%s idx=%s changes=%s",
                getattr(new_model, "backend", None),
                getattr(new_model, "index", None),
                diff,
            )

            # --- Persist validated model back BEFORE touching preview ---
            self._working_settings.cameras[row] = new_model
            self._update_active_list_item(row, new_model)

            # Decide whether we need to restart preview (fast UX)
            old_settings = None
            if self._preview.backend and isinstance(getattr(self._preview.backend, "settings", None), CameraSettings):
                old_settings = self._preview.backend.settings
            else:
                old_settings = current_model

            restart = False
            should_consider_restart = self._preview.state == PreviewState.ACTIVE and isinstance(
                old_settings, CameraSettings
            )
            if should_consider_restart:
                restart = self._should_restart_preview(old_settings, new_model)

            LOGGER.debug(
                "[Apply] preview_state=%s restart=%s backend=%s idx=%s",
                self._preview.state,
                restart,
                new_model.backend,
                new_model.index,
            )

            if self._preview.state == PreviewState.ACTIVE and restart:
                self._append_status("[Apply] Restarting preview to apply camera settings changes.")
                self._request_preview_restart(new_model, reason="apply-settings")

            self.apply_settings_btn.setEnabled(False)
            self._set_apply_dirty(False)
            return True

        except Exception as exc:
            LOGGER.exception("Apply camera settings failed")
            QMessageBox.warning(self, "Apply Settings Error", str(exc))
            return False

    def _clear_settings_form(self) -> None:
        self.cam_enabled_checkbox.setChecked(True)
        self.cam_name_label.setText("")
        self.cam_device_name_label.setText("")
        self.cam_index_label.setText("")
        self.cam_backend_label.setText("")
        self.detected_resolution_label.setText("—")
        self.detected_fps_label.setText("—")
        self.cam_width.setValue(0)
        self.cam_height.setValue(0)
        self.cam_fps.setValue(0.0)
        self.cam_exposure.setValue(0)
        self.cam_gain.setValue(0.0)
        self.cam_rotation.setCurrentIndex(0)
        self.cam_crop_x0.setValue(0)
        self.cam_crop_y0.setValue(0)
        self.cam_crop_x1.setValue(0)
        self.cam_crop_y1.setValue(0)
        self.apply_settings_btn.setEnabled(False)
        self.reset_settings_btn.setEnabled(False)

    def _populate_from_settings(self) -> None:
        """Populate the dialog from existing settings."""
        self.active_cameras_list.clear()
        for i, cam in enumerate(self._working_settings.cameras):
            item = QListWidgetItem(self._format_camera_label(cam, i))
            item.setData(Qt.ItemDataRole.UserRole, cam)
            if not cam.enabled:
                item.setForeground(Qt.GlobalColor.gray)
            self.active_cameras_list.addItem(item)

        if self.active_cameras_list.count() > 0:
            self.active_cameras_list.setCurrentRow(0)

        self._refresh_available_cameras()
        self._update_button_states()

    def _reset_selected_camera(self, *, clear_backend_cache: bool = False) -> None:
        """Reset the selected camera by probing device defaults and applying them to requested values."""
        if self._current_edit_index is None:
            return
        row = self._current_edit_index
        if row < 0 or row >= len(self._working_settings.cameras):
            return

        # Stop preview to avoid fighting an open capture
        if self._is_preview_live():
            self._stop_preview()

        cam = self._working_settings.cameras[row]

        # Set requested fields to Auto first (so backend won't force a mode)
        cam.width = 0
        cam.height = 0
        cam.fps = 0.0
        cam.exposure = 0
        cam.gain = 0.0
        cam.rotation = 0
        cam.crop_x0 = cam.crop_y0 = cam.crop_x1 = cam.crop_y1 = 0

        # Clear cached detected extras so the probe definitely runs
        if isinstance(cam.properties, dict):
            bkey = (cam.backend or "").lower()
            ns = cam.properties.get(bkey)
            if isinstance(ns, dict):
                if clear_backend_cache:
                    ns.clear()
                else:
                    ns.pop("detected_resolution", None)
                    ns.pop("detected_fps", None)
                    ns.pop("last_applied_resolution", None)

        # Update UI immediately to show "Auto" while probing
        self._load_camera_to_form(cam)
        self._append_status("[Reset] Probing device defaults…")

        # Start probe and apply detected values back to requested settings
        self._start_probe_for_camera(cam, apply_to_requested=True)

        self.apply_settings_btn.setEnabled(True)

    def _on_ok_clicked(self) -> None:
        # Auto-apply pending edits before saving
        if not self._commit_pending_edits(reason="before going back to the main window"):
            return
        if len(self._working_settings.get_active_cameras()) > self.MAX_CAMERAS:
            QMessageBox.warning(self, "Maximum Cameras", f"Maximum of {self.MAX_CAMERAS} active cameras allowed.")
            return
        try:
            if self.apply_settings_btn.isEnabled():
                self._append_status("[OK button] Auto-applying pending settings before closing dialog.")
                self._apply_camera_settings()
        except Exception:
            LOGGER.exception("[OK button] Auto-apply failed")
        self._stop_preview()
        active = self._working_settings.get_active_cameras()
        if self._working_settings.cameras and not active:
            QMessageBox.warning(self, "No Active Cameras", "Please enable at least one camera or remove all cameras.")
            return
        self._multi_camera_settings = self._working_settings.model_copy(deep=True)
        self.settings_changed.emit(self._multi_camera_settings)

        self._on_close_cleanup()
        self.accept()

    # -------------------------------
    # Probe management
    # -------------------------------

    def _start_probe_for_camera(self, cam: CameraSettings, *, apply_to_requested: bool = False) -> None:
        """Start a quick probe to fill detected labels.

        If apply_to_requested=True, the probe result will also overwrite the selected camera's
        requested width/height/fps with detected device values.
        """
        # Don’t probe if preview is active/loading
        if self._is_preview_live():
            return

        pw = getattr(self, "_probe_worker", None)
        if pw and pw.isRunning():
            try:
                pw.request_cancel()
            except Exception:
                pass
            pw.wait(200)
        self._probe_worker = None

        # Track probe intent
        self._probe_apply_to_requested = bool(apply_to_requested)
        self._probe_target_row = int(self._current_edit_index) if self._current_edit_index is not None else None

        # Show current detected values if present
        self._set_detected_labels(cam)

        # If we already have detected values and we are NOT applying them, skip probing
        backend = (cam.backend or "").lower()
        props = cam.properties if isinstance(cam.properties, dict) else {}
        ns = props.get(backend, {}) if isinstance(props.get(backend, None), dict) else {}
        if not apply_to_requested:
            det_res = ns.get("detected_resolution")
            if isinstance(det_res, (list, tuple)) and len(det_res) == 2:
                try:
                    if int(det_res[0]) > 0 and int(det_res[1]) > 0:
                        return
                except Exception:
                    pass

        # Start probe worker (settings will be opened in GUI thread for safety)
        self._probe_worker = CameraProbeWorker(cam, self)
        self._probe_worker.progress.connect(self._append_status)
        self._probe_worker.success.connect(self._on_probe_success)
        self._probe_worker.error.connect(self._on_probe_error)
        self._probe_worker.finished.connect(self._on_probe_finished)
        self._probe_worker.start()

    def _on_probe_success(self, payload) -> None:
        """Open/close quickly to read actual_resolution/actual_fps and store as detected_*.

        If self._probe_apply_to_requested is True, also overwrite requested width/height/fps
        for the targeted camera row (Reset behavior).
        """
        if not isinstance(payload, CameraSettings):
            return
        cam_settings = payload

        try:
            be = CameraFactory.create(cam_settings)
            be.open()

            actual_res = getattr(be, "actual_resolution", None)
            actual_fps = getattr(be, "actual_fps", None)

            try:
                be.close()
            except Exception:
                pass

            backend = (cam_settings.backend or "").lower()

            for i, c in enumerate(self._working_settings.cameras):
                if (c.backend or "").lower() == backend and int(c.index) == int(cam_settings.index):
                    # Ensure backend namespace exists
                    if not isinstance(c.properties, dict):
                        c.properties = {}
                    ns = c.properties.setdefault(backend, {})
                    if not isinstance(ns, dict):
                        ns = {}
                        c.properties[backend] = ns

                    # ---- Store DETECTED values (read-only telemetry) ----
                    # Store regardless of "set_*" support. This is just "what device reports".
                    if actual_res and isinstance(actual_res, (list, tuple)) and len(actual_res) == 2:
                        ns["detected_resolution"] = [int(actual_res[0]), int(actual_res[1])]
                    elif actual_res and isinstance(actual_res, tuple) and len(actual_res) == 2:
                        ns["detected_resolution"] = [int(actual_res[0]), int(actual_res[1])]

                    if isinstance(actual_fps, (int, float)) and float(actual_fps) > 0:
                        ns["detected_fps"] = float(actual_fps)
                        self._append_status(f"[Probe] actual_res={actual_res}, actual_fps={actual_fps}")

                    # ---- Apply detected -> requested (Reset behavior) ----
                    if self._probe_apply_to_requested and self._probe_target_row == i:
                        # Only apply resolution if we actually got it
                        if "detected_resolution" in ns:
                            c.width = int(ns["detected_resolution"][0])
                            c.height = int(ns["detected_resolution"][1])

                        # FPS: if device reports 0 (OpenCV often does), keep Auto (0.0)
                        if "detected_fps" in ns and float(ns["detected_fps"]) > 0:
                            c.fps = float(ns["detected_fps"])
                        else:
                            c.fps = 0.0

                        self._append_status("[Reset] Applied detected values to requested settings.")
                        if c.width > 0 and c.height > 0:
                            self._append_status(f"[Reset] Requested resolution set to {c.width}x{c.height}.")
                        if c.fps > 0:
                            self._append_status(f"[Reset] Requested FPS set to {c.fps:.2f}.")
                        else:
                            self._append_status("[Reset] Requested FPS set to Auto (device did not report FPS).")

                        # Refresh UI for current selection
                        self._load_camera_to_form(c)
                        self._update_active_list_item(i, c)

                    # Always refresh detected labels if currently selected
                    if self._current_edit_index == i:
                        self._set_detected_labels(c)
                    break

        except Exception as exc:
            self._append_status(f"[Probe] Error: {exc}")
        finally:
            self._probe_apply_to_requested = False
            self._probe_target_row = None

    def _on_probe_error(self, msg: str) -> None:
        self._append_status(f"[Probe] {msg}")

    def _on_probe_finished(self) -> None:
        self._probe_worker = None

    def _merge_backend_settings_back(self, opened_settings: CameraSettings) -> None:
        """Merge identity/index changes learned during preview open back into the working settings."""
        if self._current_edit_index is None:
            return
        row = self._current_edit_index
        if row < 0 or row >= len(self._working_settings.cameras):
            return

        target = self._working_settings.cameras[row]

        # Update index if backend rebinding occurred
        try:
            target.index = int(opened_settings.index)
        except Exception:
            pass

        if isinstance(opened_settings.properties, dict):
            if not isinstance(target.properties, dict):
                target.properties = {}
            for k, v in opened_settings.properties.items():
                if isinstance(v, dict) and isinstance(target.properties.get(k), dict):
                    target.properties[k].update(v)
                else:
                    target.properties[k] = v

        # Update UI list item text to reflect any changes
        self._update_active_list_item(row, target)

    def _adjust_preview_timer_for_fps(self, fps: float | None) -> None:
        """Adjust preview cadence to match actual FPS (bounded for CPU)."""
        if not self._preview.timer or not fps or fps <= 0:
            return
        interval_ms = max(15, int(1000.0 / min(max(fps, 1.0), 60.0)))
        self._preview.timer.start(interval_ms)

    def _reconcile_fps_from_backend(self, cam: CameraSettings) -> None:
        """Reconcile preview cadence to actual FPS without overriding Auto request."""

        # If user requested Auto (0), do not overwrite the request.
        if float(getattr(cam, "fps", 0.0) or 0.0) <= 0.0:
            actual = getattr(self._preview.backend, "actual_fps", None) if self._preview.backend else None
            if actual:
                self._append_status(f"[Info] Auto FPS; device reports ~{actual:.2f}. Preview timer adjusted.")
                self._adjust_preview_timer_for_fps(actual)
            else:
                self._append_status("[Info] Auto FPS; backend can't reliably report actual FPS.")
            return

        # If user requested a specific FPS, optionally clamp UI to actual if measurable.
        actual = getattr(self._preview.backend, "actual_fps", None) if self._preview.backend else None
        if actual is None:
            self._append_status("[Info] Backend can't reliably report actual FPS; keeping requested value.")
            return

        if abs(cam.fps - actual) > 0.5:
            cam.fps = actual
            self.cam_fps.setValue(actual)
            self._append_status(f"[Info] FPS adjusted to device-supported ~{actual:.2f}.")
            self._adjust_preview_timer_for_fps(actual)
        else:
            self._adjust_preview_timer_for_fps(actual)

    # ---------------------------------
    # Preview lifecycle management (start/stop + loading state)
    # ---------------------------------
    def _toggle_preview(self) -> None:
        if self._preview.state == PreviewState.LOADING:
            self._cancel_loading()
            return
        if self._preview.state == PreviewState.ACTIVE:
            self._stop_preview()
        else:
            self._start_preview()

    def _begin_preview_load(self, cam: CameraSettings, *, reason: str) -> None:
        """
        Begin (re)loading preview for cam.

        Purpose:
        - Bumps epoch of the preview to invalidate
          any in-flight loader results from previous previews.
          See _bump_epoch and _on_loader_* methods.
        - Enters LOADING state
        - Creates and wires loader
        - Sets requested_cam
        """
        LOGGER.debug("[Preview] begin load reason=%s backend=%s idx=%s", reason, cam.backend, cam.index)

        # If already loading, just coalesce restart/intention
        if self._preview.state == PreviewState.LOADING:
            self._preview.pending_restart = copy.deepcopy(cam)
            return

        # Stop any existing backend/timer/loader cleanly
        self._stop_preview_internal(reason="begin-load")

        self._preview.state = PreviewState.LOADING
        epoch = self._bump_epoch()
        self._preview.requested_cam = copy.deepcopy(cam)
        self._preview.pending_restart = None
        self._preview.restart_scheduled = False

        # Force preview-safe backend flags
        if isinstance(self._preview.requested_cam.properties, dict):
            ns = self._preview.requested_cam.properties.setdefault((cam.backend or "").lower(), {})
            if isinstance(ns, dict):
                ns["fast_start"] = False

        loader = CameraLoadWorker(self._preview.requested_cam, self)
        self._preview.loader = loader

        # Connect signals with epoch captured
        loader.progress.connect(lambda msg, e=epoch: self._on_loader_progress(e, msg))
        loader.success.connect(lambda payload, e=epoch: self._on_loader_success(e, payload))
        loader.error.connect(lambda err, e=epoch: self._on_loader_error(e, err))
        loader.canceled.connect(lambda e=epoch: self._on_loader_canceled(e))
        loader.finished.connect(lambda e=epoch: self._on_loader_finished(e))

        # UI
        self.preview_status.clear()
        self._show_loading_overlay("Loading camera…")
        self._sync_preview_ui()

        loader.start()

    def _start_preview(self) -> None:
        """Start camera preview asynchronously (no UI freeze)."""
        if not self._commit_pending_edits(reason="before starting preview"):
            return
        if self._is_preview_live():
            return

        row = self._current_edit_index
        if row is None or row < 0:
            row = self.active_cameras_list.currentRow()

        if row is None or row < 0:
            LOGGER.warning("[Preview] No camera selected to start preview.")
            return

        self._current_edit_index = row
        LOGGER.debug(
            "[Preview] resolved start row=%s active_row=%s",
            self._current_edit_index,
            self.active_cameras_list.currentRow(),
        )

        item = self.active_cameras_list.item(self._current_edit_index)
        if not item:
            return
        cam = item.data(Qt.ItemDataRole.UserRole)
        if not cam:
            return

        self._begin_preview_load(cam, reason="user-start")

    def _stop_preview(self) -> None:
        self._stop_preview_internal(reason="user-stop")
        self._sync_preview_ui()

    def _stop_preview_internal(self, *, reason: str) -> None:
        """Tear down loader/backend/timer. Safe to call from anywhere."""
        LOGGER.debug("[Preview] stop reason=%s state=%s", reason, self._preview.state)

        self._preview.state = PreviewState.STOPPING

        # Invalidate all in-flight signals immediately
        self._bump_epoch()

        # Cancel loader
        if self._preview.loader and self._preview.loader.isRunning():
            self._preview.loader.request_cancel()
            self._preview.loader.wait(1500)
        self._preview.loader = None

        # Stop timer
        if self._preview.timer:
            self._preview.timer.stop()
        self._preview.timer = None

        # Close backend
        if self._preview.backend:
            try:
                self._preview.backend.close()
            except Exception:
                pass
        self._preview.backend = None
        self._preview.pending_restart = None
        self._preview.requested_cam = None
        self._preview.restart_scheduled = False

        self._hide_loading_overlay()
        self._preview.state = PreviewState.IDLE

    def _bump_epoch(self) -> int:
        self._preview.epoch += 1
        return self._preview.epoch

    def _should_restart_preview(self, old: CameraSettings, new: CameraSettings) -> bool:
        """
        Fast UX policy:
        - Do NOT restart for rotation/crop (preview applies those live).
        - Restart for camera-side capture params: resolution/fps/exposure/gain.
        Backend-agnostic for now (no OpenCV special casing).
        """
        # Restart on these changes
        for key in ("width", "height", "fps", "exposure", "gain"):
            try:
                if getattr(old, key, None) != getattr(new, key, None):
                    return True
            except Exception:
                return True  # safest: restart

        # No restart needed if only rotation/crop/enabled changed
        return False

    def _request_preview_restart(self, cam: CameraSettings, *, reason: str) -> None:
        """
        Request a preview restart. Coalesced to at most one scheduled callback.
        If currently LOADING, stores pending_restart instead of thrashing loader.
        """
        if self._preview.state == PreviewState.LOADING:
            self._preview.pending_restart = copy.deepcopy(cam)
            return

        if self._preview.state != PreviewState.ACTIVE:
            return

        self._preview.pending_restart = copy.deepcopy(cam)

        if self._preview.restart_scheduled:
            return
        self._preview.restart_scheduled = True

        QTimer.singleShot(0, lambda: self._execute_pending_restart(reason=reason))

    def _execute_pending_restart(self, *, reason: str) -> None:
        self._preview.restart_scheduled = False
        cam = self._preview.pending_restart
        self._preview.pending_restart = None
        if not cam:
            return

        LOGGER.debug("[Preview] executing restart reason=%s", reason)
        self._begin_preview_load(cam, reason="restart")

    def _cancel_loading(self) -> None:
        loader = self._preview.loader
        if loader and loader.isRunning():
            self._append_status("Cancel requested…")
            loader.request_cancel()
        else:
            # If nothing is running, ensure state is IDLE
            self._stop_preview_internal(reason="cancel-loading-noop")
        self._sync_preview_ui()

    def _is_current_epoch(self, e: int) -> bool:
        return e == self._preview.epoch

    # Loader signal handlers
    def _on_loader_progress(self, e: int, message: str) -> None:
        if not self._is_current_epoch(e):
            return
        self._show_loading_overlay(message)
        self._append_status(message)

    def _on_loader_success(self, e: int, payload) -> None:
        if not self._is_current_epoch(e):
            return
        if self._preview.state != PreviewState.LOADING:
            return

        try:
            if not isinstance(payload, CameraSettings):
                raise TypeError(f"Unexpected success payload type: {type(payload)}")
            self._append_status("Opening camera…")
            LOGGER.debug(
                "[Loader] success -> opening camera backend=%s idx=%s props_keys=%s",
                payload.backend,
                payload.index,
                list(payload.properties.keys()) if isinstance(payload.properties, dict) else None,
            )
            backend = CameraFactory.create(payload)
            backend.open()
            self._preview.backend = backend
            self._preview.state = PreviewState.ACTIVE

            req_w = getattr(self._preview.backend.settings, "width", None)
            req_h = getattr(self._preview.backend.settings, "height", None)
            actual_res = getattr(self._preview.backend, "actual_resolution", None)
            if req_w and req_h:
                if actual_res:
                    self._append_status(
                        f"Requested resolution: {req_w}x{req_h}, actual: {actual_res[0]}x{actual_res[1]}"
                    )
                else:
                    self._append_status(f"Requested resolution: {req_w}x{req_h}, actual: unknown")

            opened_sttngs = getattr(self._preview.backend, "settings", None)
            if isinstance(opened_sttngs, CameraSettings):
                backend = opened_sttngs.backend
                index = opened_sttngs.index
                device_name = (opened_sttngs.properties or {}).get(backend.lower(), {}).get("device_name", "")
                msg = f"Opened {backend}:{index}"
                if device_name:
                    msg += f" ({device_name})"
                self._append_status(msg)
                self._merge_backend_settings_back(opened_sttngs)
                if self._current_edit_index is not None and 0 <= self._current_edit_index < len(
                    self._working_settings.cameras
                ):
                    self._load_camera_to_form(self._working_settings.cameras[self._current_edit_index])
                self._reconcile_fps_from_backend(opened_sttngs)

            # Start preview UX
            self._hide_loading_overlay()
            self._sync_preview_ui()

            # Timer @ ~25 fps default; cadence may be overridden above
            self._preview.timer = QTimer(self)
            self._preview.timer.timeout.connect(self._update_preview)
            self._preview.timer.start(40)

            # FPS reconciliation + cadence (single source of truth)
            actual_fps = getattr(self._preview.backend, "actual_fps", None)
            if actual_fps:
                self._adjust_preview_timer_for_fps(actual_fps)

            self.apply_settings_btn.setEnabled(True)
        except Exception as exc:
            self._on_loader_error(e, str(exc))

    def _on_loader_error(self, e: int, error: str) -> None:
        if not self._is_current_epoch(e):
            return
        self._append_status(f"Error: {error}")
        LOGGER.error("[Loader] error: %s", error)
        self._stop_preview_internal(reason="loader-error")
        QMessageBox.warning(self, "Preview Error", f"Failed to start camera preview:\n{error}")
        self._sync_preview_ui()

    def _on_loader_canceled(self, e: int) -> None:
        if not self._is_current_epoch(e):
            return
        self._append_status("Loading canceled.")
        self._stop_preview_internal(reason="loader-canceled")
        self._sync_preview_ui()

    def _on_loader_finished(self, e: int) -> None:
        if not self._is_current_epoch(e):
            return

        pending = self._preview.pending_restart
        self._preview.pending_restart = None
        self._preview.restart_scheduled = False
        self._preview.loader = None

        if pending and self._preview.state == PreviewState.IDLE:
            LOGGER.debug("[Loader] finished with pending restart for backend=%s idx=%s", pending.backend, pending.index)
            self._begin_preview_load(pending, reason="pending-restart-after-finish")

        self._sync_preview_ui()

    def _update_preview(self) -> None:
        """Update preview frame."""
        if self._preview.state != PreviewState.ACTIVE or not self._preview.backend:
            return

        try:
            frame, _ = self._preview.backend.read()
            if frame is None or frame.size == 0:
                return

            # Apply rotation if set in the form (real-time from UI)
            rotation = self.cam_rotation.currentData()
            frame = apply_rotation(frame, rotation)

            # Compute crop with clamping
            h, w = frame.shape[:2]
            x0 = max(0, min(self.cam_crop_x0.value(), w))
            y0 = max(0, min(self.cam_crop_y0.value(), h))
            x1_val = self.cam_crop_x1.value()
            y1_val = self.cam_crop_y1.value()
            x1 = max(0, min(x1_val if x1_val > 0 else w, w))
            y1 = max(0, min(y1_val if y1_val > 0 else h, h))

            # Only apply if valid rectangle; otherwise skip crop
            if x1 > x0 and y1 > y0:
                frame = apply_crop(frame, x0, y0, x1, y1)
            else:
                # Optional: show a status once, not every frame
                pass

            # Resize to fit preview label
            frame = resize_to_fit(frame, max_w=400, max_h=300)

            q_img = to_display_pixmap(frame)
            self.preview_label.setPixmap(q_img)

        except Exception as exc:
            LOGGER.debug(f"Preview frame skipped: {exc}")
