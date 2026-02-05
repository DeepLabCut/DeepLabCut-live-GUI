"""OpenCV command-line camera discovery utility. For development/testing."""

# dlclivegui/cameras/backends/utils/opencv_cli.py
from __future__ import annotations

import argparse

import cv2

from dlclivegui.cameras.backends.utils.opencv_discovery import list_cameras


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="ANY", help="CAP_* backend (e.g. DSHOW, MSMF, AVFOUNDATION, V4L2, ANY)")
    args = p.parse_args()

    backend = getattr(cv2, f"CAP_{args.backend.upper()}", cv2.CAP_ANY)
    cams = list_cameras(backend)

    if not cams:
        print("No cameras found (or cv2-enumerate-cameras not installed).")
        return 1

    for c in cams:
        print(f"- {c.name} | idx={c.index} backend={c.backend} path={c.path} vid={c.vid} pid={c.pid} id={c.stable_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
