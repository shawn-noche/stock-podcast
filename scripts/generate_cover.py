#!/usr/bin/env python3
"""One-off utility: generates docs/cover.jpg podcast artwork (3000x3000,
meets Apple/Spotify requirements). Re-run manually if you want to
regenerate the art after changing config.py.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "docs", "cover.jpg")
SIZE = 3000

BG_TOP = np.array([10, 14, 26])
BG_BOTTOM = np.array([16, 28, 34])
RADAR_GREEN = (66, 245, 170)
TEXT_WHITE = (240, 245, 240)
TEXT_MUTED = (140, 200, 180)

FONT_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT_REGULAR_CANDIDATES = [
    "/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(candidates, size):
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_gradient_background():
    ramp = np.linspace(0, 1, SIZE).reshape(SIZE, 1, 1)
    grad = BG_TOP.reshape(1, 1, 3) * (1 - ramp) + BG_BOTTOM.reshape(1, 1, 3) * ramp
    grad = np.repeat(grad, SIZE, axis=1).astype(np.uint8)
    return Image.fromarray(grad, mode="RGB")


def draw_radar(draw, center, max_radius, n_rings=5):
    cx, cy = center
    for i in range(1, n_rings + 1):
        r = max_radius * i / n_rings
        alpha = max(20, 90 - i * 12)
        color = RADAR_GREEN + (alpha,)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=6)

    # Sweep wedge
    sweep = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    sweep_draw = ImageDraw.Draw(sweep)
    sweep_draw.pieslice(
        [cx - max_radius, cy - max_radius, cx + max_radius, cy + max_radius],
        start=-95, end=-40, fill=RADAR_GREEN + (60,),
    )
    return sweep


def main():
    base = make_gradient_background().convert("RGBA")
    overlay = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    center = (SIZE * 0.5, SIZE * 0.62)
    max_radius = SIZE * 0.42
    sweep = draw_radar(draw, center, max_radius, n_rings=6)
    base = Image.alpha_composite(base, overlay)
    base = Image.alpha_composite(base, sweep)

    # A single bright "blip" dot on one ring, like a detected contact.
    blip_draw = ImageDraw.Draw(base)
    blip_angle = math.radians(-65)
    blip_r = max_radius * 0.68
    bx = center[0] + blip_r * math.cos(blip_angle)
    by = center[1] + blip_r * math.sin(blip_angle)
    blip_draw.ellipse([bx - 22, by - 22, bx + 22, by + 22], fill=RADAR_GREEN + (255,))
    blip_draw.ellipse([bx - 46, by - 46, bx + 46, by + 46], outline=RADAR_GREEN + (140,), width=4)

    title_font = load_font(FONT_BOLD_CANDIDATES, 260)
    subtitle_font = load_font(FONT_REGULAR_CANDIDATES, 90)

    title_lines = ["UNDER THE", "RADAR"]
    y = SIZE * 0.08
    for line in title_lines:
        bbox = blip_draw.textbbox((0, 0), line, font=title_font)
        w = bbox[2] - bbox[0]
        blip_draw.text(((SIZE - w) / 2, y), line, font=title_font, fill=TEXT_WHITE)
        y += (bbox[3] - bbox[1]) + 60

    subtitle = "STOCKS NOBODY'S TALKING ABOUT YET"
    bbox = blip_draw.textbbox((0, 0), subtitle, font=subtitle_font)
    w = bbox[2] - bbox[0]
    blip_draw.text(((SIZE - w) / 2, y + 70), subtitle, font=subtitle_font, fill=TEXT_MUTED)

    final = base.convert("RGB")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    final.save(OUT_PATH, "JPEG", quality=92)
    print(f"Wrote {OUT_PATH} ({SIZE}x{SIZE})")


if __name__ == "__main__":
    main()
