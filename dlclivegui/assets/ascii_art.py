"""
Utilities to generate ASCII (optionally ANSI-colored) art for the user's terminal.

Cross-platform and CI-safe:
- Detects terminal width using shutil.get_terminal_size (portable across OSes).
- Respects NO_COLOR and a color mode (auto|always|never).
- Enables ANSI color on Windows PowerShell/cmd via os.system("") when needed.
- Supports transparent PNGs (alpha) by compositing over a chosen background color.
- Optional crop-to-content using alpha or a background heuristic when no alpha.

Dependencies: opencv-python, numpy
"""

# dlclivegui/assets/ascii_art.py
from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Iterable
from typing import Literal

import numpy as np

from dlclivegui.gui.theme import LOGO_ALPHA

try:
    import cv2 as cv
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "OpenCV (opencv-python) is required for dlclivegui.assets.ascii_art.\nInstall with: pip install opencv-python"
    ) from e

# Character ramps (dense -> sparse)
ASCII_RAMP_SIMPLE = "@%#*+=-:. "
ASCII_RAMP_FINE = "@$B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

ColorMode = Literal["auto", "always", "never"]

# -----------------------------
# Terminal / ANSI capabilities
# -----------------------------


def enable_windows_ansi_support() -> None:
    """Enable ANSI escape support in Windows terminals.
    Safe to call on any OS; no-op on non-Windows.
    """
    if os.name == "nt":
        # This call toggles the console mode to enable VT processing in many hosts
        #  Always leave the string empty.
        os.system("")  # This is a known, safe workaround to enable ANSI support on Windows.


def get_terminal_width(default: int = 80) -> int:
    """Return terminal width in columns, or a fallback if stdout is not a TTY."""
    try:
        if not sys.stdout.isatty():
            return default
        return shutil.get_terminal_size((default, 24)).columns
    except Exception:
        return default


def should_use_color(mode: ColorMode = "auto") -> bool:
    """Determine if colored ANSI output should be emitted.

    - 'never': never use color
    - 'always': always use color (even when redirected)
    - 'auto': use color only when stdout is a TTY and NO_COLOR is not set
    """
    if mode == "never":
        return False
    if mode == "always":
        return True
    # auto
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def terminal_is_wide_enough(min_width: int = 60) -> bool:
    if not sys.stdout.isatty():
        return False
    return get_terminal_width() >= min_width


# -----------------------------
# Image helpers
# -----------------------------


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure an image array is 3-channel BGR."""
    if img.ndim == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        # Caller should composite first; keep as-is for now
        b, g, r, a = cv.split(img)
        return cv.merge((b, g, r))
    raise ValueError(f"Unsupported image shape for BGR conversion: {img.shape!r}")


def composite_over_color(img: np.ndarray, bg_bgr: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """If img has alpha (BGRA), alpha-composite over a solid BGR color and return BGR."""
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv.split(img)
        af = (a.astype(np.float32) / 255.0)[..., None]  # (H,W,1)
        bgr = cv.merge((b, g, r)).astype(np.float32)
        bg = np.empty_like(bgr, dtype=np.float32)
        bg[..., 0] = bg_bgr[0]
        bg[..., 1] = bg_bgr[1]
        bg[..., 2] = bg_bgr[2]
        out = af * bgr + (1.0 - af) * bg
        return np.clip(out, 0, 255).astype(np.uint8)
    return _to_bgr(img)


def crop_to_content_alpha(img_bgra: np.ndarray, alpha_thresh: int = 1, pad: int = 0) -> np.ndarray:
    """Crop to bounding box of pixels where alpha > alpha_thresh. Returns BGRA."""
    if not (img_bgra.ndim == 3 and img_bgra.shape[2] == 4):
        return img_bgra
    a = img_bgra[..., 3]
    mask = a > alpha_thresh
    if not mask.any():
        return img_bgra
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    if pad:
        h, w = a.shape
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(h - 1, y1 + pad)
        x1 = min(w - 1, x1 + pad)
    return img_bgra[y0 : y1 + 1, x0 : x1 + 1, :]


def crop_to_content_bg(
    img_bgr: np.ndarray, bg: Literal["white", "black"] = "white", tol: int = 10, pad: int = 0
) -> np.ndarray:
    """Heuristic crop when no alpha: assume uniform white or black background.
    Returns BGR.
    """
    if not (img_bgr.ndim == 3 and img_bgr.shape[2] == 3):
        img_bgr = _to_bgr(img_bgr)
    if bg == "white":
        dist = 255 - img_bgr.max(axis=2)  # darker than white
        mask = dist > tol
    else:
        dist = img_bgr.max(axis=2)  # brighter than black
        mask = dist > tol
    if not mask.any():
        return img_bgr
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    if pad:
        h, w = mask.shape
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(h - 1, y1 + pad)
        x1 = min(w - 1, x1 + pad)
    return img_bgr[y0 : y1 + 1, x0 : x1 + 1, :]


def resize_for_terminal(img: np.ndarray, width: int | None, aspect: float | None) -> np.ndarray:
    """Resize image for terminal display.

    Parameters
    ----------
    width: target character width (None -> current terminal width)
    aspect: character cell height/width ratio; default 0.5 is good for many fonts.
    """
    h, w = img.shape[:2]
    if width is None:
        width = get_terminal_width(100)
    width = max(20, int(width))
    if aspect is None:
        # Allow override by env var, else default 0.5
        try:
            aspect = float(os.environ.get("DLCLIVE_ASCII_ASPECT", "0.5"))
        except ValueError:
            aspect = 0.5
    new_h = max(1, int((h / w) * width * aspect))
    return cv.resize(img, (width, new_h), interpolation=cv.INTER_AREA)


# -----------------------------
# Rendering
# -----------------------------


def _map_luminance_to_chars(gray: np.ndarray, fine: bool) -> Iterable[str]:
    ramp = ASCII_RAMP_FINE if fine else ASCII_RAMP_SIMPLE
    ramp_arr = np.array(list(ramp), dtype="<U1")  # vectorized char LUT

    idx = (gray.astype(np.float32) / 255.0 * (len(ramp) - 1)).astype(np.int32)
    chars = ramp_arr[idx]  # (H,W) array of 1-char strings

    # Join per-row (still Python per row, but NOT per pixel)
    return ["".join(row.tolist()) for row in chars]


def _color_ascii_lines(img_bgr: np.ndarray, fine: bool, invert: bool) -> Iterable[str]:
    ramp = ASCII_RAMP_FINE if fine else ASCII_RAMP_SIMPLE
    # ramp is ASCII; encode once
    ramp_bytes = [c.encode("utf-8") for c in ramp]

    reset = b"\x1b[0m"

    # Luminance (same coefficients you used; keep exact behavior)
    b = img_bgr[..., 0].astype(np.float32)
    g = img_bgr[..., 1].astype(np.float32)
    r = img_bgr[..., 2].astype(np.float32)
    lum = 0.0722 * b + 0.7152 * g + 0.2126 * r
    if invert:
        lum = 255.0 - lum

    idx = (lum / 255.0 * (len(ramp) - 1)).astype(np.uint16)

    # Pack color into 0xRRGGBB for fast comparisons
    rr = img_bgr[..., 2].astype(np.uint32)
    gg = img_bgr[..., 1].astype(np.uint32)
    bb = img_bgr[..., 0].astype(np.uint32)
    color_key = (rr << 16) | (gg << 8) | bb  # (H,W) uint32

    # Cache SGR prefixes by packed color
    # e.g. 0xRRGGBB -> b"\x1b[38;2;R;G;Bm"
    prefix_cache: dict[int, bytes] = {}

    h, w = idx.shape
    lines: list[str] = []

    for y in range(h):
        ba = bytearray()

        ck_row = memoryview(color_key[y])
        idx_row = memoryview(idx[y])

        prev_ck: int | None = None

        for x in range(w):
            ck = int(ck_row[x])

            # Emit new color code only when color changes
            if ck != prev_ck:
                prefix = prefix_cache.get(ck)
                if prefix is None:
                    rr_i = (ck >> 16) & 255
                    gg_i = (ck >> 8) & 255
                    bb_i = ck & 255
                    prefix = f"\x1b[38;2;{rr_i};{gg_i};{bb_i}m".encode("ascii")
                    prefix_cache[ck] = prefix
                ba.extend(prefix)
                prev_ck = ck

            ba.extend(ramp_bytes[int(idx_row[x])])

        # Reset once per line to prevent color bleed into subsequent terminal output
        ba.extend(reset)

        lines.append(ba.decode("utf-8", errors="strict"))

    return lines


# -----------------------------
# Public API
# -----------------------------


def generate_ascii_lines(
    image_path: str,
    *,
    width: int | None = None,
    aspect: float | None = None,
    color: ColorMode = "auto",
    fine: bool = False,
    invert: bool = False,
    crop_content: bool = False,
    crop_bg: Literal["none", "white", "black"] = "none",
    alpha_thresh: int = 1,
    crop_pad: int = 0,
    bg_bgr: tuple[int, int, int] = (255, 255, 255),
) -> Iterable[str]:
    """Load an image and return ASCII art lines sized for the user's terminal.

    Parameters
    ----------
    image_path: path to the input image
    width: output width in characters (None -> detect terminal width)
    aspect: character cell height/width ratio (None -> 0.5 or env override)
    color: 'auto'|'always'|'never' color mode
    fine: use a finer 70+ character ramp
    invert: invert luminance mapping
    crop_content: crop to non-transparent content (alpha) if present
    crop_bg: when no alpha, optionally crop assuming a uniform 'white' or 'black' background
    alpha_thresh: threshold for alpha-based crop (0-255)
    crop_pad: pixels of padding around detected content
    bg_bgr: background color used for alpha compositing (default white)
    """
    enable_windows_ansi_support()

    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    # Load preserving alpha if present
    img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to load image with OpenCV: {image_path}")

    # Crop prior to compositing/resizing
    if crop_content and img.ndim == 3 and img.shape[2] == 4:
        img = crop_to_content_alpha(img, alpha_thresh=alpha_thresh, pad=crop_pad)
    elif crop_content and (img.ndim != 3 or img.shape[2] != 4) and crop_bg in ("white", "black"):
        img = crop_to_content_bg(_to_bgr(img), bg=crop_bg, tol=10, pad=crop_pad)

    # Composite transparency to solid background for correct visual result
    img_bgr = composite_over_color(img, bg_bgr=bg_bgr)

    # Resize for terminal cell ratio
    img_bgr = resize_for_terminal(img_bgr, width=width, aspect=aspect)

    use_color = should_use_color(color)

    if use_color:
        return _color_ascii_lines(img_bgr, fine=fine, invert=invert)
    else:
        gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        if invert:
            gray = 255 - gray
        return _map_luminance_to_chars(gray, fine=fine)


def print_ascii(
    image_path: str,
    *,
    width: int | None = None,
    aspect: float | None = None,
    color: ColorMode = "auto",
    fine: bool = False,
    invert: bool = False,
    crop_content: bool = False,
    crop_bg: Literal["none", "white", "black"] = "none",
    alpha_thresh: int = 1,
    crop_pad: int = 0,
    bg_bgr: tuple[int, int, int] = (255, 255, 255),
    output: str | None = None,
) -> None:
    """Convenience: generate and print ASCII art; optionally write it to a file."""
    lines = list(
        generate_ascii_lines(
            image_path,
            width=width,
            aspect=aspect,
            color=color,
            fine=fine,
            invert=invert,
            crop_content=crop_content,
            crop_bg=crop_bg,
            alpha_thresh=alpha_thresh,
            crop_pad=crop_pad,
            bg_bgr=bg_bgr,
        )
    )

    # Print to stdout
    for line in lines:
        print(line)

    # Optionally write raw ANSI/plain text to a file
    if output:
        with open(output, "w", encoding="utf-8", newline="\n") as f:
            for line in lines:
                f.write(line)
                f.write("\n")


# -----------------------------
# Optional: Help banner helpers
# -----------------------------


def build_help_description(
    static_banner: str | None = None, *, desc=None, color: ColorMode = "auto", min_width: int = 60, max_width: int = 120
) -> str:
    """Return a help description string that conditionally includes a colored ASCII banner.

    - If stdout is a TTY and wide enough, returns banner + description.
    - Otherwise returns a plain, single-line description.
    - If static_banner is None, uses ASCII_BANNER (empty by default).
    """
    enable_windows_ansi_support()
    desc = "DeepLabCut-Live GUI — launch the graphical interface." if desc is None else desc
    if not sys.stdout.isatty() and terminal_is_wide_enough(min_width=min_width):
        return desc

    banner: str | None
    if static_banner is not None:
        banner = static_banner
    else:
        try:
            term_width = get_terminal_width(default=max_width)
            width = max(min(term_width, max_width), min_width)
            banner = "\n".join(
                generate_ascii_lines(
                    str(LOGO_ALPHA),
                    width=width,
                    aspect=0.5,
                    color=color,
                    fine=True,
                    invert=False,
                    crop_content=True,
                    crop_bg="white",
                    alpha_thresh=1,
                    crop_pad=1,
                    bg_bgr=(255, 255, 255),
                )
            )
        except (FileNotFoundError, RuntimeError, OSError):
            banner = None

    if banner:
        if should_use_color(color):
            banner = f"\x1b[36m{banner}\x1b[0m"
        return banner + "\n" + desc
    return desc
