# dlclivegui/gui/camera_config/preview.py
import cv2
from PySide6.QtGui import QImage, QPixmap

# def _update_preview(self) -> None:
#     """Update preview frame."""
#     if self._preview.state != PreviewState.ACTIVE or not self._preview.backend:
#         return

#     try:
#         frame, _ = self._preview.backend.read()
#         if frame is None or frame.size == 0:
#             return

#         # Apply rotation if set in the form (real-time from UI)
#         rotation = self.cam_rotation.currentData()
#         if rotation == 90:
#             frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#         elif rotation == 180:
#             frame = cv2.rotate(frame, cv2.ROTATE_180)
#         elif rotation == 270:
#             frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

#         # Apply crop if set in the form (real-time from UI)
#         h, w = frame.shape[:2]
#         x0 = self.cam_crop_x0.value()
#         y0 = self.cam_crop_y0.value()
#         x1 = self.cam_crop_x1.value() or w
#         y1 = self.cam_crop_y1.value() or h
#         # Clamp to frame bounds
#         x0 = max(0, min(x0, w))
#         y0 = max(0, min(y0, h))
#         x1 = max(x0, min(x1, w))
#         y1 = max(y0, min(y1, h))
#         if x1 > x0 and y1 > y0:
#             frame = frame[y0:y1, x0:x1]

#         # Resize to fit preview label
#         h, w = frame.shape[:2]
#         max_w, max_h = 400, 300
#         scale = min(max_w / w, max_h / h)
#         new_w, new_h = int(w * scale), int(h * scale)
#         frame = cv2.resize(frame, (new_w, new_h))

#         # Convert to QImage and display
#         if frame.ndim == 2:
#             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#         elif frame.shape[2] == 4:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
#         else:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         h, w, ch = frame.shape
#         bytes_per_line = ch * w
#         q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
#         self.preview_label.setPixmap(QPixmap.fromImage(q_img))

#     except Exception as exc:
#         LOGGER.debug(f"Preview frame skipped: {exc}")


def apply_rotation(frame, rotation):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def apply_crop(frame, x0, y0, x1, y1):
    h, w = frame.shape[:2]
    x0 = max(0, min(x0, w))
    y0 = max(0, min(y0, h))
    x1 = max(x0, min(x1, w))
    y1 = max(y0, min(y1, h))

    if x1 > x0 and y1 > y0:
        return frame[y0:y1, x0:x1]
    return frame


def resize_to_fit(frame, max_w=400, max_h=300):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h))


def to_display_pixmap(frame):
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w, ch = frame.shape
    bytes_per_line = ch * w
    q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(q_img)
