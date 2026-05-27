"""
Build uniform, legible qualitative panels for the report.

The raw annotated images have mismatched aspect ratios (two portrait, one
landscape), lots of empty background, and thin/pale boxes. This script
re-renders each from its saved Spatial Evidence Graph: crop to the object
region, draw thick bright boxes + centroid line, and emit all panels at a
single uniform size so the figure reads as a clean grid.

Output: results/figures/qual_panel_{1,2,3}.jpg
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
QUAL = ROOT / "results" / "qualitative"
OUT = ROOT / "results" / "figures"

TARGET_W, TARGET_H = 760, 570          # uniform 4:3 panels
TARGET_AR = TARGET_W / TARGET_H
PAD_FRAC = 0.12                         # background margin around the object union
GREEN = (40, 220, 90)                   # obj1
RED = (255, 60, 60)                     # obj2
LINE = (250, 210, 30)                   # centroid connector

SCENES = [
    ("Left_Right", "qual_panel_1.jpg"),
    ("Inside_Outside", "qual_panel_2.jpg"),
    ("Behind_Infront", "qual_panel_3.jpg"),
]


def load_seg(scene: str) -> dict:
    return json.loads((QUAL / scene / "result_q1.json").read_text())


def expand_to_aspect(box, W, H):
    """Expand a pixel box to TARGET_AR, re-centring and clamping to the image."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw / bh < TARGET_AR:
        bw = bh * TARGET_AR
    else:
        bh = bw / TARGET_AR
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1, x2 = cx - bw / 2, cx + bw / 2
    y1, y2 = cy - bh / 2, cy + bh / 2
    # shift inside bounds where possible
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > W: x1 -= (x2 - W); x2 = W
    if y2 > H: y1 -= (y2 - H); y2 = H
    return [max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))]


def render(scene: str, out_name: str):
    seg = load_seg(scene)
    ev = next((e for e in seg["evidence"] if e.get("obj1_bbox") and e.get("obj2_bbox")), None)
    img = Image.open(QUAL / scene / "orignal_image.jpg").convert("RGB")
    W, H = img.size
    if ev is None:
        crop_box = [0, 0, W, H]
        b1 = b2 = None
    else:
        b1, b2 = ev["obj1_bbox"], ev["obj2_bbox"]
        ux1 = min(b1["x1"], b2["x1"]); uy1 = min(b1["y1"], b2["y1"])
        ux2 = max(b1["x2"], b2["x2"]); uy2 = max(b1["y2"], b2["y2"])
        # to pixels + padding
        px = PAD_FRAC * W; py = PAD_FRAC * H
        box = [ux1 * W - px, uy1 * H - py, ux2 * W + px, uy2 * H + py]
        crop_box = expand_to_aspect(box, W, H)

    crop = img.crop(crop_box)
    cw, ch = crop.size
    draw = ImageDraw.Draw(crop)
    lw = max(5, cw // 130)

    def to_crop(bx):
        return [bx["x1"] * W - crop_box[0], bx["y1"] * H - crop_box[1],
                bx["x2"] * W - crop_box[0], bx["y2"] * H - crop_box[1]]

    if b1 and b2:
        r1, r2 = to_crop(b1), to_crop(b2)
        c1 = ((r1[0] + r1[2]) / 2, (r1[1] + r1[3]) / 2)
        c2 = ((r2[0] + r2[2]) / 2, (r2[1] + r2[3]) / 2)
        draw.line([c1, c2], fill=LINE, width=max(3, lw // 2))
        draw.rectangle(r1, outline=GREEN, width=lw)
        draw.rectangle(r2, outline=RED, width=lw)
        for c, col in ((c1, GREEN), (c2, RED)):
            rdot = lw
            draw.ellipse([c[0] - rdot, c[1] - rdot, c[0] + rdot, c[1] + rdot], fill=col)

    panel = crop.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    panel.save(OUT / out_name, "JPEG", quality=92)
    print(f"wrote {OUT / out_name}  ({seg['obj1']} / {seg['obj2']}, {seg['relation']})")


def main():
    for scene, out_name in SCENES:
        render(scene, out_name)


if __name__ == "__main__":
    main()
