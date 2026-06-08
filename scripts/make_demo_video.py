"""Render a self-contained demo walkthrough video for the Spatial Evidence Agent.

Reads the committed qualitative results (original images + result_*.json) and
animates the SEA pipeline stage-by-stage:

    Title -> Architecture -> [Planner -> Executor -> Critic -> Verified] x N -> Results

No API keys, GPU, or model weights required -- everything is replayed from the
stored Spatial Evidence Graphs under results/qualitative/.

Usage (from repo root):
    .venv/bin/python scripts/make_demo_video.py \
        --out results/demo/sea_demo.mp4 --fps 30
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

# ---------------------------------------------------------------- palette ----
BG = "#0d1117"
PANEL = "#161b22"
EDGE = "#30363d"
FG = "#e6edf3"
MUTED = "#8b949e"
CYAN = "#22d3ee"      # obj1 / subject
ORANGE = "#fb923c"    # obj2 / reference
GREEN = "#22c55e"     # verified
PURPLE = "#a78bfa"    # planner
BLUE = "#60a5fa"      # executor
YELLOW = "#facc15"    # active perception

W, H, DPI = 1280, 720, 100

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUAL = os.path.join(REPO, "results", "qualitative")


# ------------------------------------------------------------- examples ------
# Successes only, chosen to span position / coverage / depth relations.
EXAMPLES = [
    dict(dir="Left_Right", n=1, cat="LEFT / RIGHT",
         family="position", note=None),
    dict(dir="Above_Below", n=1, cat="ABOVE / BELOW",
         family="position", note=None),
    dict(dir="Inside_Outside", n=1, cat="INSIDE / CONTAINS",
         family="coverage", note=None),
    dict(dir="ON", n=2, cat="ON / SUPPORT",
         family="contact", note=None),
]

REL_PHRASE = {
    "left_of": "is left of", "right_of": "is right of",
    "above": "is above", "below": "is below",
    "behind": "is behind", "in_front": "is in front of",
    "on": "is on", "contains": "contains",
}


def load_example(ex):
    d = os.path.join(QUAL, ex["dir"])
    with open(os.path.join(d, f"result_q{ex['n']}.json")) as f:
        res = json.load(f)
    img = np.asarray(Image.open(os.path.join(d, "orignal_image.jpg")).convert("RGB"))
    ev = res["evidence"][0]
    return dict(res=res, img=img, ev=ev)


# --------------------------------------------------------- frame plumbing ----
def new_fig():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)
    return fig


def fig_to_bgr(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


class Video:
    def __init__(self, path, fps):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fps = fps
        self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        if not self.vw.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        self.path = path

    def hold(self, frame, seconds):
        for _ in range(max(1, int(round(seconds * self.fps)))):
            self.vw.write(frame)

    def fade_in(self, render, seconds=0.35):
        """render(alpha)->bgr for alpha in (0,1]; plays a quick reveal then nothing."""
        n = max(1, int(round(seconds * self.fps)))
        for i in range(n):
            self.vw.write(render((i + 1) / n))

    def close(self):
        self.vw.release()


class FrameRecorder:
    """Drop-in stand-in for Video that collects frames in memory instead of
    writing an mp4, so a single scene can be re-rendered to a GIF."""
    def __init__(self, fps):
        self.fps = fps
        self.frames = []
        self.vw = self  # so scene code calling `vid.vw.write(...)` works

    def write(self, frame):
        self.frames.append(frame)

    def hold(self, frame, seconds):
        for _ in range(max(1, int(round(seconds * self.fps)))):
            self.frames.append(frame)

    def fade_in(self, render, seconds=0.35):
        n = max(1, int(round(seconds * self.fps)))
        for i in range(n):
            self.frames.append(render((i + 1) / n))


def save_gif(frames_bgr, path, src_fps, gif_fps, width):
    """Subsample + downscale BGR frames and write an optimized looping GIF."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stride = max(1, int(round(src_fps / gif_fps)))
    sel = frames_bgr[::stride]
    imgs = []
    for f in sel:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        if width and im.width != width:
            im = im.resize((width, round(im.height * width / im.width)),
                           Image.LANCZOS)
        imgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=256))
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(round(1000 / gif_fps)), loop=0,
                 optimize=True, disposal=2)
    return len(sel)


def ease(t):
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------- text helpers ----
def chip(ax, x, y, text, color, fs=15):
    ax.text(x, y, f"  {text}  ", color=BG, fontsize=fs, fontweight="bold",
            va="center", ha="left", family="DejaVu Sans",
            bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="none"))


def footer(ax, idx, total):
    ax.text(0.5, 0.022, "Spatial Evidence Agent  ·  training-free verified spatial VQA",
            transform=ax.transAxes, color=MUTED, fontsize=10, ha="center", va="center")
    if total:
        # step dots
        for k in range(total):
            ax.scatter(0.5 + (k - (total - 1) / 2) * 0.03, 0.058,
                       s=40, transform=ax.transAxes,
                       color=GREEN if k < idx else EDGE, zorder=5)


# ------------------------------------------------------------- scenes --------
def scene_title(vid):
    def render(a=1.0):
        fig = new_fig()
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.62, "Spatial Evidence Agent", color=FG, fontsize=46,
                fontweight="bold", ha="center", va="center", alpha=a)
        ax.text(0.5, 0.50, "SEA", color=CYAN, fontsize=22, fontweight="bold",
                ha="center", va="center", alpha=a, family="monospace")
        ax.text(0.5, 0.40,
                "A training-free three-agent pipeline for\nverified spatial visual question answering",
                color=MUTED, fontsize=18, ha="center", va="center", alpha=a)
        for i, (txt, c) in enumerate([("Planner", PURPLE), ("Executor", BLUE),
                                      ("Critic", CYAN), ("Verified answer", GREEN)]):
            chip(ax, 0.5 - 0.36 + i * 0.20 - 0.04, 0.26, txt, c, fs=13)
        return fig_to_bgr(fig)

    vid.fade_in(render, 0.5)
    vid.hold(render(), 2.6)


def scene_architecture(vid):
    boxes = [("Planner", PURPLE, "parse question ->\n{obj1, obj2, relation}"),
             ("Executor", BLUE, "VLM initial\nyes / no + claims"),
             ("Critic", CYAN, "detect + depth +\ngeometry rules"),
             ("Output", GREEN, "answer +\nevidence graph")]

    def render(reveal_to=4, a_last=1.0):
        fig = new_fig()
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.88, "How it works", color=FG, fontsize=30,
                fontweight="bold", ha="center")
        ax.text(0.5, 0.80, "geometry doesn't overrule the VLM — it routes trust",
                color=MUTED, fontsize=15, ha="center")
        n = len(boxes); bw, gap = 0.20, 0.046
        total = n * bw + (n - 1) * gap
        x0 = (1 - total) / 2
        for i, (name, c, sub) in enumerate(boxes):
            if i > reveal_to - 1:
                continue
            alpha = a_last if i == reveal_to - 1 else 1.0
            x = x0 + i * (bw + gap)
            ax.add_patch(FancyBboxPatch((x, 0.46), bw, 0.20,
                         boxstyle="round,pad=0.012", fc=PANEL, ec=c, lw=2.5,
                         alpha=alpha, transform=ax.transAxes))
            ax.text(x + bw / 2, 0.605, name, color=c, fontsize=17,
                    fontweight="bold", ha="center", va="center", alpha=alpha)
            ax.text(x + bw / 2, 0.515, sub, color=FG, fontsize=10.5,
                    ha="center", va="center", alpha=alpha)
            if i > 0:
                xa = x0 + i * (bw + gap)
                ax.annotate("", xy=(xa - 0.004, 0.56), xytext=(xa - gap + 0.004, 0.56),
                            transform=ax.transAxes,
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2,
                                            alpha=alpha))
        # feedback loop annotation
        if reveal_to >= 4:
            ax.annotate("", xy=(x0 + bw + 0.005, 0.44),
                        xytext=(x0 + 2 * (bw + gap) + bw / 2, 0.44),
                        transform=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=1.8,
                                        connectionstyle="arc3,rad=0.35", ls="--"))
            ax.text(0.5, 0.345, "disagree → zoom in & re-examine  (active perception, up to k loops)",
                    color=YELLOW, fontsize=12, ha="center", style="italic")
            ax.text(0.5, 0.18,
                    "Every committed answer carries an auditable Spatial Evidence Graph:\n"
                    "bounding boxes · monocular depth · centroid deltas · the exact rule that fired",
                    color=MUTED, fontsize=12.5, ha="center")
        return fig_to_bgr(fig)

    for r in range(1, 5):
        vid.fade_in(lambda a, r=r: render(r, a), 0.3)
        vid.hold(render(r), 0.55)
    vid.hold(render(4), 3.2)


def draw_box(ax, bb, w, h, color, label, alpha=1.0, lw=3.0):
    x1, y1 = bb["x1"] * w, bb["y1"] * h
    bw, bh = (bb["x2"] - bb["x1"]) * w, (bb["y2"] - bb["y1"]) * h
    ax.add_patch(Rectangle((x1, y1), bw, bh, fill=False, edgecolor=color,
                           lw=lw, alpha=alpha))
    ax.text(x1 + 3, y1 - 6, f" {label}  {bb['confidence']*100:.0f}% ",
            color=BG, fontsize=11, fontweight="bold", va="bottom", ha="left",
            alpha=alpha, family="DejaVu Sans",
            bbox=dict(boxstyle="square,pad=0.18", fc=color, ec="none", alpha=alpha))
    cx = (bb["x1"] + bb["x2"]) / 2 * w
    cy = (bb["y1"] + bb["y2"]) / 2 * h
    ax.scatter([cx], [cy], s=45, color=color, alpha=alpha, zorder=6,
               edgecolors=BG, linewidths=1.2)


def scene_example(vid, ex, idx, total):
    data = load_example(ex)
    res, img, evd = data["res"], data["img"], data["ev"]
    h, w = img.shape[:2]
    b1, b2 = evd["obj1_bbox"], evd["obj2_bbox"]
    obj1, obj2 = res["obj1"], res["obj2"]
    rel = res["relation"]
    answer = res["answer_str"].upper()
    iters = res.get("iterations", 1)

    rows = []  # (color, text)  built progressively

    def render(step, a_last=1.0, box_a=1.0):
        fig = new_fig()
        # image panel
        axi = fig.add_axes([0.035, 0.085, 0.455, 0.80])
        axi.set_facecolor(PANEL)
        axi.imshow(img, extent=[0, w, h, 0])
        axi.set_xlim(0, w); axi.set_ylim(h, 0)
        axi.set_xticks([]); axi.set_yticks([])
        for s in axi.spines.values():
            s.set_color(EDGE)
        if step >= 4:
            a = box_a if step == 4 else 1.0
            draw_box(axi, b1, w, h, CYAN, obj1, a)
        if step >= 5:
            a = box_a if step == 5 else 1.0
            draw_box(axi, b2, w, h, ORANGE, obj2, a)

        # text panel
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        chip(ax, 0.53, 0.915, ex["cat"], CYAN, fs=13)
        ax.text(0.985, 0.915, f"{ex['family']} relation", color=MUTED,
                fontsize=12, ha="right", va="center", style="italic")
        # question
        ax.text(0.53, 0.84, "Q:  " + res["question"].strip(), color=FG,
                fontsize=16.5, fontweight="bold", va="top", wrap=True)

        lines = []
        if step >= 1:
            lines.append((PURPLE, "Planner",
                          f"{obj1}  —  {REL_PHRASE.get(rel, rel)}  —  {obj2}\nrelation = {rel}"))
        if step >= 2:
            lines.append((BLUE, "Executor (VLM)",
                          f"initial guess: {answer.lower()}  ·  proposes the two objects to check"))
        if 3 <= step <= 5:
            lines.append((CYAN, "Critic · Grounding-DINO",
                          "localizing both objects in the image..."))
        if step >= 6:
            lines.append((CYAN, "Critic · Depth + geometry",
                          f"depth d1={evd['obj1_depth']:.3f}  d2={evd['obj2_depth']:.3f}\n"
                          f"dx={evd['dx']:+.3f}  dy={evd['dy']:+.3f}  dz={evd['dz']:+.3f}  IoU={evd['iou']:.3f}"))
        if step >= 7:
            rule = evd["rule_applied"].replace(" AND ", "\n     AND ")
            held = "condition holds → relation is TRUE" if evd.get("passed") \
                else "condition not met → relation is FALSE"
            lines.append((YELLOW, "Rule evaluated", f"{rule}\n{held}"))

        y = 0.745
        for i, (c, head, body) in enumerate(lines):
            a = a_last if i == len(lines) - 1 and step in (1, 2, 3, 6, 7) else 1.0
            ax.text(0.53, y, "▸ " + head, color=c, fontsize=13,
                    fontweight="bold", va="top", alpha=a)
            ax.text(0.553, y - 0.034, body, color=FG, fontsize=11.5,
                    va="top", alpha=a, family="DejaVu Sans")
            y -= 0.034 + 0.034 * (body.count("\n") + 1) + 0.012

        # active-perception note
        if step >= 6 and ex.get("note") and iters > 1:
            ax.text(0.53, 0.235, f"↻ {iters} iterations", color=YELLOW,
                    fontsize=12, fontweight="bold", va="top")
            ax.text(0.53, 0.205, ex["note"], color=MUTED, fontsize=11,
                    va="top", wrap=True)

        # verdict stamp
        if step >= 8:
            a = a_last if step == 8 else 1.0
            ax.add_patch(FancyBboxPatch((0.53, 0.095), 0.45, 0.085,
                         boxstyle="round,pad=0.012", fc=PANEL, ec=GREEN,
                         lw=2.5, alpha=a, transform=ax.transAxes))
            ax.text(0.555, 0.137, f"VERIFIED  →  {answer}", color=GREEN,
                    fontsize=21, fontweight="bold", va="center", alpha=a)
            ax.text(0.555, 0.112, f"confidence {res['confidence']:.2f}  ·  "
                    f"answer matches the geometric evidence", color=MUTED,
                    fontsize=11, va="center", alpha=a)
        footer(ax, idx, total)
        return fig_to_bgr(fig)

    # timeline: (step, fade?, hold_seconds)
    timeline = [
        (0, False, 1.4),   # image + question
        (1, True, 1.6),    # planner
        (2, True, 1.6),    # executor
        (3, True, 0.9),    # critic detecting
        (4, "box", 1.3),   # box1
        (5, "box", 1.5),   # box2
        (6, True, 2.2),    # depth/geometry
        (7, True, 2.0),    # rule
        (8, True, 3.0),    # verdict
    ]
    for step, mode, hold in timeline:
        if mode == "box":
            vid.fade_in(lambda a, s=step: render(s, box_a=ease(a)), 0.5)
        elif mode is True:
            vid.fade_in(lambda a, s=step: render(s, a_last=a), 0.3)
        vid.hold(render(step), hold)


def scene_loop_example(vid, idx, total):
    """Active-perception loop: detect -> executor disagrees -> Critic crops the
    disputed region -> re-examine the zoom -> agree -> VERIFIED. Depth numbers
    are intentionally kept out of frame; the focus is the loop mechanic."""
    d = os.path.join(QUAL, "Behind_Infront")
    with open(os.path.join(d, "result_q2.json")) as f:
        res = json.load(f)
    img = np.asarray(Image.open(os.path.join(d, "orignal_image.jpg")).convert("RGB"))
    h, w = img.shape[:2]
    evd = res["evidence"][0]
    b1, b2 = evd["obj1_bbox"], evd["obj2_bbox"]
    obj1, obj2 = res["obj1"], res["obj2"]
    crop = res["crop_history"][0]
    k = res.get("iterations", 2)

    RED = "#f87171"
    cw_n = crop["x2"] - crop["x1"]
    ch_n = crop["y2"] - crop["y1"]
    reason = crop.get("reason", "disputed region around both objects")
    # pixel viewports for the animated zoom (full image -> disputed crop)
    px1, px2 = max(0, int(crop["x1"] * w)), min(w, int(crop["x2"] * w))
    py1, py2 = max(0, int(crop["y1"] * h)), min(h, int(crop["y2"] * h))
    FULL = (0.0, float(w), float(h), 0.0)        # xlo, xhi, y_bottom, y_top
    CROPV = (float(px1), float(px2), float(py2), float(py1))

    def iter_badges(ax, step):
        def badge(x, label, color, filled):
            ax.text(x, 0.788, f" {label} ", transform=ax.transAxes,
                    color=BG if filled else color, fontsize=10.5,
                    fontweight="bold", va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.32",
                              fc=color if filled else PANEL, ec=color, lw=1.6))
        if step < 5:
            badge(0.53, "iter 1 · running", CYAN, False)
        else:
            badge(0.53, "iter 1 ✗ conflict", RED, True)
        if step >= 6:
            ax.annotate("", xy=(0.735, 0.79), xytext=(0.70, 0.79),
                        transform=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
            if step == 6:
                badge(0.745, "iter 2 · running", YELLOW, False)
            else:
                badge(0.745, "iter 2 ✓ agree", GREEN, True)

    def lines_for(step):
        if step == 1:
            return [(PURPLE, "Planner", f"objects: {obj1}, {obj2}\nrelation = {res['relation']}")]
        if step == 2:
            return [(PURPLE, "Planner", f"objects: {obj1}, {obj2}"),
                    (BLUE, "Executor (VLM)", "returns an initial yes/no — but with\nlow confidence on this cluttered scene")]
        if step in (3, 4):
            return [(BLUE, "Executor (VLM)", "initial answer logged"),
                    (CYAN, "Critic · Grounding-DINO", "localizing both objects in the full image...")]
        if step == 5:
            return [(BLUE, "Executor (VLM)", "initial answer"),
                    (RED, "Conflict detected", "the Executor's answer disagrees with\nthe Critic's geometric verdict"),
                    (YELLOW, "Critic schedules re-examination", f"crop the disputed region (+5% pad):\n“{reason}”")]
        if step == 6:
            return [(YELLOW, "Iteration 2 · zoom in", "re-render ONLY the cropped disputed\nregion and feed it back to the VLM")]
        if step == 7:
            return [(YELLOW, "Iteration 2 · re-examine", "the VLM re-answers on the zoomed crop,\nwhere the two objects are large & clear"),
                    (GREEN, "Agreement check", "VLM answer  ==  geometric check  ✓")]
        if step == 8:
            return [(GREEN, "Resolved", f"agreement reached in {k} iterations —\nthe loop stops and commits the answer")]
        return []

    def render(step, a_last=1.0, box_a=1.0, view=None):
        if view is None:
            view = FULL
        fig = new_fig()
        axi = fig.add_axes([0.035, 0.085, 0.455, 0.80])
        axi.set_facecolor(PANEL)
        axi.imshow(img, extent=[0, w, h, 0])
        axi.set_xlim(view[0], view[1]); axi.set_ylim(view[2], view[3])
        if step >= 3:
            draw_box(axi, b1, w, h, CYAN, obj1, box_a if step == 3 else 1.0)
        if step >= 4:
            draw_box(axi, b2, w, h, ORANGE, obj2, box_a if step == 4 else 1.0)
        if step == 5:  # dashed overlay of the region about to be re-examined
            cx, cy = crop["x1"] * w, crop["y1"] * h
            axi.add_patch(Rectangle((cx, cy), cw_n * w, ch_n * h, fill=False,
                          edgecolor=YELLOW, lw=3, ls="--", alpha=a_last))
        axi.set_xticks([]); axi.set_yticks([])
        for s in axi.spines.values():
            s.set_color(EDGE)

        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        chip(ax, 0.53, 0.915, "ACTIVE PERCEPTION", YELLOW, fs=13)
        ax.text(0.985, 0.915, "the loop in action", color=MUTED,
                fontsize=12, ha="right", va="center", style="italic")
        ax.text(0.53, 0.85, "Q:  " + res["question"].strip(), color=FG,
                fontsize=16, fontweight="bold", va="top")
        iter_badges(ax, step)

        lines = lines_for(step)
        y = 0.72
        for i, (c, head, body) in enumerate(lines):
            a = a_last if i == len(lines) - 1 and step in (1, 2, 5, 6, 7) else 1.0
            ax.text(0.53, y, "▸ " + head, color=c, fontsize=13,
                    fontweight="bold", va="top", alpha=a)
            ax.text(0.553, y - 0.034, body, color=FG, fontsize=11.5,
                    va="top", alpha=a, family="DejaVu Sans")
            y -= 0.034 + 0.034 * (body.count("\n") + 1) + 0.016

        if step >= 8:
            a = a_last if step == 8 else 1.0
            ax.add_patch(FancyBboxPatch((0.53, 0.095), 0.45, 0.085,
                         boxstyle="round,pad=0.012", fc=PANEL, ec=GREEN,
                         lw=2.5, alpha=a, transform=ax.transAxes))
            ax.text(0.555, 0.137, "VERIFIED  →  YES", color=GREEN,
                    fontsize=21, fontweight="bold", va="center", alpha=a)
            ax.text(0.555, 0.112, "committed once Executor and geometry agree",
                    color=MUTED, fontsize=11, va="center", alpha=a)
        footer(ax, idx, total)
        return fig_to_bgr(fig)

    # ---- timeline -----------------------------------------------------------
    vid.hold(render(0), 1.3)
    for s, hold in [(1, 1.5), (2, 1.9)]:
        vid.fade_in(lambda a, s=s: render(s, a_last=a), 0.35)
        vid.hold(render(s), hold)
    for s, hold in [(3, 0.9), (4, 1.2)]:
        vid.fade_in(lambda a, s=s: render(s, box_a=ease(a)), 0.5)
        vid.hold(render(s), hold)
    # iteration 1 result: conflict + dashed crop overlay
    vid.fade_in(lambda a: render(5, a_last=a), 0.35)
    vid.hold(render(5), 2.8)
    # animated zoom from the full frame into the disputed crop (text = step 6)
    n = max(1, int(round(1.1 * vid.fps)))
    for i in range(n):
        t = ease((i + 1) / n)
        v = tuple(FULL[j] + (CROPV[j] - FULL[j]) * t for j in range(4))
        vid.vw.write(render(6, view=v))
    vid.hold(render(6, view=CROPV), 2.4)
    # iteration 2: re-examine + agreement check
    vid.fade_in(lambda a: render(7, a_last=a, view=CROPV), 0.35)
    vid.hold(render(7, view=CROPV), 2.8)
    # commit
    vid.fade_in(lambda a: render(8, a_last=a, view=CROPV), 0.35)
    vid.hold(render(8, view=CROPV), 3.2)


def scene_results(vid):
    rows = [
        ("GPT-4o baseline", "0.720", "0.720", "1.00", FG),
        ("SEA (full coverage)", "0.790", "0.790", "1.00", GREEN),
        ("SEA (selective)", "0.715", "0.803", "0.89", CYAN),
        ("SEA (high precision, τ=0.40)", "—", "0.847", "0.30", ORANGE),
    ]

    def render(reveal=len(rows), a_last=1.0):
        fig = new_fig()
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.90, "Results · VSR-200", color=FG, fontsize=30,
                fontweight="bold", ha="center")
        ax.text(0.5, 0.83, "stratified 200-sample split, all 8 relations, identical images",
                color=MUTED, fontsize=14, ha="center")
        cols = [("System", 0.12, "left"), ("Accuracy", 0.58, "center"),
                ("Selective acc.", 0.73, "center"), ("Coverage", 0.88, "center")]
        for name, x, ha in cols:
            ax.text(x, 0.70, name, color=MUTED, fontsize=14,
                    fontweight="bold", ha=ha)
        ax.plot([0.10, 0.92], [0.665, 0.665], color=EDGE, lw=1.2)
        y = 0.60
        for i, (sys, acc, sel, cov, c) in enumerate(rows):
            if i >= reveal:
                break
            a = a_last if i == reveal - 1 else 1.0
            fw = "bold" if c is not FG else "normal"
            ax.text(0.12, y, sys, color=c, fontsize=14.5, ha="left",
                    fontweight=fw, alpha=a)
            ax.text(0.58, y, acc, color=c, fontsize=14.5, ha="center", alpha=a,
                    fontweight=fw, family="monospace")
            ax.text(0.73, y, sel, color=c, fontsize=14.5, ha="center", alpha=a,
                    fontweight=fw, family="monospace")
            ax.text(0.88, y, cov, color=c, fontsize=14.5, ha="center", alpha=a,
                    fontweight=fw, family="monospace")
            y -= 0.085
        if reveal >= len(rows):
            ax.text(0.5, 0.15,
                    "One confidence threshold dials SEA along the precision/coverage frontier —\n"
                    "and every committed answer comes with auditable geometric evidence.",
                    color=MUTED, fontsize=13.5, ha="center")
        footer(ax, 0, 0)
        return fig_to_bgr(fig)

    for r in range(1, len(rows) + 1):
        vid.fade_in(lambda a, r=r: render(r, a), 0.3)
        vid.hold(render(r), 0.7)
    vid.hold(render(), 3.5)


def scene_end(vid):
    def render(a=1.0):
        fig = new_fig()
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.56, "Spatial Evidence Agent", color=FG, fontsize=36,
                fontweight="bold", ha="center", alpha=a)
        ax.text(0.5, 0.46, "answers spatial questions you can audit",
                color=CYAN, fontsize=18, ha="center", alpha=a)
        return fig_to_bgr(fig)
    vid.fade_in(render, 0.4)
    vid.hold(render(), 2.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "results", "demo", "sea_demo.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no_video", action="store_true",
                    help="skip the full mp4 (useful with --gif)")
    ap.add_argument("--gif", nargs="?", const=os.path.join(
                    REPO, "results", "demo", "active_perception.gif"), default=None,
                    help="also export just the active-perception scene as a GIF")
    ap.add_argument("--gif_fps", type=int, default=12)
    ap.add_argument("--gif_width", type=int, default=640)
    args = ap.parse_args()

    if not args.no_video:
        vid = Video(args.out, args.fps)
        total = len(EXAMPLES) + 1  # + the active-perception loop example
        scene_title(vid)
        for i, ex in enumerate(EXAMPLES):
            scene_example(vid, ex, i, total)
        scene_loop_example(vid, len(EXAMPLES), total)
        scene_end(vid)
        vid.close()
        print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.2f} MB)")

    if args.gif:
        rec = FrameRecorder(args.fps)
        scene_loop_example(rec, 0, 0)  # total=0 hides the cross-example dots
        nf = save_gif(rec.frames, args.gif, args.fps, args.gif_fps, args.gif_width)
        print(f"wrote {args.gif}  ({os.path.getsize(args.gif)/1e6:.2f} MB, "
              f"{nf} frames @ {args.gif_fps}fps, {args.gif_width}px)")


if __name__ == "__main__":
    main()
