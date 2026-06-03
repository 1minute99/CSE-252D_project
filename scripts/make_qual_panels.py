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
VSR_IMG = ROOT / "data" / "vsr_images"
VSR_EVIDENCE = ROOT / "results" / "vsr200_evidence_recal_k2.json"
VSR_SPLIT = ROOT / "data" / "vsr_strat200.json"
OUT = ROOT / "results" / "figures"

TARGET_W, TARGET_H = 760, 570          # uniform 4:3 panels
TARGET_AR = TARGET_W / TARGET_H
PAD_FRAC = 0.12                         # background margin around the object union
GREEN = (40, 220, 90)                   # obj1
RED = (255, 60, 60)                     # obj2
LINE = (250, 210, 30)                   # centroid connector

# Each scene is either:
#   ("qual",  scene_dir, seg_name, out_name)              from results/qualitative/
#   ("vsr",   vsr_idx,                out_name)           from VSR-200 evidence dump
SCENES = [
    ("qual", "Left_Right",     "result_q1.json", "qual_panel_1.jpg"),  # left_of
    ("vsr",  50,                                 "qual_panel_2.jpg"),  # right_of
    ("qual", "Above_Below",    "result_q1.json", "qual_panel_3.jpg"),  # above
    ("vsr",  73,                                 "qual_panel_4.jpg"),  # below
    ("qual", "Inside_Outside", "result_q1.json", "qual_panel_5.jpg"),  # contains
    ("qual", "ON",             "result_q1.json", "qual_panel_6.jpg"),  # on (verified no)
    ("vsr",  31,                                 "qual_panel_7.jpg"),  # in_front
    ("vsr",  127,                                "qual_panel_8.jpg"),  # behind (verified)
    ("qual", "Behind_Infront", "result_q1.json", "qual_panel_9.jpg"),  # behind (abstained)
    # SEA-vs-baseline panels: cases where GPT-4o alone failed but the Critic
    # produced the correct verdict (used in Figure 11 of the report).
    ("vsr",  163,                                "sea_vs_baseline_1.jpg"),  # left_of override (skateboard/dog)
    ("vsr",  192,                                "sea_vs_baseline_2.jpg"),  # above override (umbrella/cat)
    ("vsr",  81,                                 "sea_vs_baseline_3.jpg"),  # right_of override, negative (person/cake)
]

# Cache VSR lookup tables on first use.
_vsr_cache: dict = {}


def _load_vsr_tables():
    if "ev" in _vsr_cache:
        return
    _vsr_cache["ev"] = {e["idx"]: e for e in json.loads(VSR_EVIDENCE.read_text())}
    _vsr_cache["split"] = json.loads(VSR_SPLIT.read_text())


def load_qual(scene: str, seg_name: str) -> tuple[Path, dict]:
    """Returns (image_path, SEG-shaped dict)."""
    seg = json.loads((QUAL / scene / seg_name).read_text())
    return QUAL / scene / "orignal_image.jpg", seg


def load_vsr(idx: int) -> tuple[Path, dict]:
    """Adapt a VSR-200 evidence entry to the SEG shape this script consumes."""
    _load_vsr_tables()
    ev = _vsr_cache["ev"][idx]
    split = _vsr_cache["split"][idx]
    img_path = VSR_IMG / split["image_path"]
    seg = {
        "obj1": ev["obj1"],
        "obj2": ev["obj2"],
        "relation": ev["relation"],
        "evidence": [{
            "obj1_bbox": ev["b1"],
            "obj2_bbox": ev["b2"],
        }],
    }
    return img_path, seg


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


def render(img_path: Path, seg: dict, out_name: str):
    ev = next((e for e in seg["evidence"] if e.get("obj1_bbox") and e.get("obj2_bbox")), None)
    img = Image.open(img_path).convert("RGB")
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
    for entry in SCENES:
        kind = entry[0]
        if kind == "qual":
            _, scene, seg_name, out_name = entry
            img_path, seg = load_qual(scene, seg_name)
        elif kind == "vsr":
            _, idx, out_name = entry
            img_path, seg = load_vsr(idx)
        else:
            raise ValueError(f"unknown scene kind: {kind}")
        render(img_path, seg, out_name)


if __name__ == "__main__":
    main()
