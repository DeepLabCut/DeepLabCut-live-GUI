# dlclivegui/gui/display/utils.py
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPixmap


def frame_to_pixmap(
    frame_bgr: np.ndarray,
    target_size: QSize,
) -> QPixmap:
    """Convert a BGR image to a smoothly scaled Qt pixmap."""
    frame = np.asarray(frame_bgr)

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Display frame must have shape (H, W, 3); received {frame.shape!r}.")

    if frame.dtype != np.uint8:
        raise ValueError(f"Display frame must use uint8 pixels; received {frame.dtype}.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape

    image = QImage(
        rgb.data,
        width,
        height,
        channels * width,
        QImage.Format.Format_RGB888,
    )

    pixmap = QPixmap.fromImage(image)

    return pixmap.scaled(
        target_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
