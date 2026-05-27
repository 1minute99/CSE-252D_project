"""
Geometry unit tests for the Critic.

Covers BoundingBox.iou, _map_bbox_to_original, and every branch of
_verify_relation. Runs with pytest (`pytest tests/`) or as a script
(`python tests/test_critic.py`).

No model dependencies — uses synthetic boxes/depths.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Make `src/` importable since the project uses flat imports.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import CriticConfig  # noqa: E402
from critic import _compute_crop, _map_bbox_to_original, _verify_relation  # noqa: E402
from state import BoundingBox, CropRegion  # noqa: E402


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, abs_tol=tol)


# ── BoundingBox.iou ──────────────────────────────────────────────────────────

def test_iou_identical_boxes_is_one():
    b = BoundingBox(x1=0.1, y1=0.1, x2=0.4, y2=0.4)
    assert _close(b.iou(b), 1.0)


def test_iou_disjoint_boxes_is_zero():
    a = BoundingBox(x1=0.0, y1=0.0, x2=0.2, y2=0.2)
    b = BoundingBox(x1=0.5, y1=0.5, x2=0.7, y2=0.7)
    assert _close(a.iou(b), 0.0)


def test_iou_half_overlap_along_x():
    # Two equal-area boxes whose intersection is half of each → IoU = 1/3.
    a = BoundingBox(x1=0.0, y1=0.0, x2=0.4, y2=0.4)
    b = BoundingBox(x1=0.2, y1=0.0, x2=0.6, y2=0.4)
    assert _close(a.iou(b), 1 / 3)


def test_iou_zero_area_box_is_zero():
    a = BoundingBox(x1=0.2, y1=0.2, x2=0.2, y2=0.4)  # zero width
    b = BoundingBox(x1=0.0, y1=0.0, x2=0.5, y2=0.5)
    assert _close(a.iou(b), 0.0)


# ── _map_bbox_to_original ────────────────────────────────────────────────────

def test_map_bbox_no_crop_is_identity():
    local = BoundingBox(x1=0.2, y1=0.3, x2=0.5, y2=0.6)
    mapped = _map_bbox_to_original(local, None)
    assert (mapped.x1, mapped.y1, mapped.x2, mapped.y2) == (0.2, 0.3, 0.5, 0.6)


def test_map_bbox_full_crop_is_identity():
    local = BoundingBox(x1=0.2, y1=0.3, x2=0.5, y2=0.6)
    full = CropRegion(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    mapped = _map_bbox_to_original(local, full)
    assert all(
        _close(a, b) for a, b in zip(
            (mapped.x1, mapped.y1, mapped.x2, mapped.y2),
            (0.2, 0.3, 0.5, 0.6),
        )
    )


def test_map_bbox_corner_crop_scales_and_offsets():
    # Crop is the top-left quarter of the image. A box covering the whole
    # crop maps to the top-left quarter of the original frame.
    local = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    crop = CropRegion(x1=0.0, y1=0.0, x2=0.5, y2=0.5)
    mapped = _map_bbox_to_original(local, crop)
    assert all(
        _close(a, b) for a, b in zip(
            (mapped.x1, mapped.y1, mapped.x2, mapped.y2),
            (0.0, 0.0, 0.5, 0.5),
        )
    )


def test_map_bbox_preserves_label_and_confidence():
    local = BoundingBox(x1=0.0, y1=0.0, x2=0.5, y2=0.5, confidence=0.91, label="cup")
    crop = CropRegion(x1=0.2, y1=0.4, x2=0.8, y2=1.0)
    mapped = _map_bbox_to_original(local, crop)
    assert mapped.label == "cup"
    assert _close(mapped.confidence, 0.91)


# ── _verify_relation: positive cases ─────────────────────────────────────────

def _b(x1, y1, x2, y2):
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def test_left_of_passes_when_obj1_clearly_left():
    cfg = CriticConfig()
    passed, _ = _verify_relation("left_of", _b(0.1, 0.4, 0.3, 0.6), _b(0.6, 0.4, 0.8, 0.6), 0.5, 0.5, cfg)
    assert passed is True


def test_right_of_passes_when_obj1_clearly_right():
    cfg = CriticConfig()
    passed, _ = _verify_relation("right_of", _b(0.6, 0.4, 0.8, 0.6), _b(0.1, 0.4, 0.3, 0.6), 0.5, 0.5, cfg)
    assert passed is True


def test_above_passes_when_obj1_higher_in_image():
    cfg = CriticConfig()
    # Smaller y means closer to top of image.
    passed, _ = _verify_relation("above", _b(0.4, 0.1, 0.6, 0.3), _b(0.4, 0.6, 0.6, 0.8), 0.5, 0.5, cfg)
    assert passed is True


def test_below_passes_when_obj1_lower_in_image():
    cfg = CriticConfig()
    passed, _ = _verify_relation("below", _b(0.4, 0.6, 0.6, 0.8), _b(0.4, 0.1, 0.6, 0.3), 0.5, 0.5, cfg)
    assert passed is True


def test_behind_passes_when_obj1_deeper():
    cfg = CriticConfig()
    # Larger depth value == farther in Depth Anything's normalization.
    passed, _ = _verify_relation("behind", _b(0.4, 0.4, 0.6, 0.6), _b(0.4, 0.4, 0.6, 0.6), 0.8, 0.2, cfg)
    assert passed is True


def test_in_front_passes_when_obj1_closer():
    cfg = CriticConfig()
    passed, _ = _verify_relation("in_front", _b(0.4, 0.4, 0.6, 0.6), _b(0.4, 0.4, 0.6, 0.6), 0.1, 0.8, cfg)
    assert passed is True


def test_on_requires_above_AND_iou():
    cfg = CriticConfig()
    # Obj1 sits on top of obj2 with overlap.
    obj1 = _b(0.40, 0.30, 0.60, 0.55)
    obj2 = _b(0.35, 0.50, 0.65, 0.80)
    passed, ev = _verify_relation("on", obj1, obj2, 0.5, 0.5, cfg)
    assert passed is True
    assert "IoU" in ev["rule_applied"]


def test_on_fails_when_above_but_no_iou():
    cfg = CriticConfig()
    obj1 = _b(0.10, 0.10, 0.30, 0.30)  # above but separate
    obj2 = _b(0.60, 0.60, 0.80, 0.80)
    passed, _ = _verify_relation("on", obj1, obj2, 0.5, 0.5, cfg)
    assert passed is False


def test_contains_passes_when_obj2_inside_obj1():
    cfg = CriticConfig()
    big = _b(0.10, 0.10, 0.90, 0.90)
    small = _b(0.40, 0.40, 0.55, 0.55)
    passed, _ = _verify_relation("contains", big, small, 0.5, 0.5, cfg)
    assert passed is True


def test_contains_fails_when_areas_similar():
    cfg = CriticConfig()
    # Even with full coverage of obj2 by obj1, near-equal areas means "contains"
    # is too strong a claim. Default area_ratio_threshold=0.70 catches this.
    a = _b(0.10, 0.10, 0.90, 0.90)
    b = _b(0.12, 0.12, 0.88, 0.88)  # area ratio ~0.92
    passed, _ = _verify_relation("contains", a, b, 0.5, 0.5, cfg)
    assert passed is False


def test_unknown_relation_returns_false_with_marker():
    cfg = CriticConfig()
    passed, ev = _verify_relation("entangled_with", _b(0, 0, 1, 1), _b(0, 0, 1, 1), 0.5, 0.5, cfg)
    assert passed is False
    assert ev["rule_applied"].startswith("unknown relation")


# ── _verify_relation: threshold sensitivity (proves config plumbing) ─────────

def test_borderline_left_of_responds_to_margin():
    """A small horizontal separation passes at margin=0.02 but fails at 0.10."""
    a = _b(0.45, 0.40, 0.50, 0.60)
    b = _b(0.50, 0.40, 0.55, 0.60)  # cx(a) - cx(b) = -0.05
    lax = _verify_relation("left_of", a, b, 0.5, 0.5, CriticConfig(margin=0.02))[0]
    strict = _verify_relation("left_of", a, b, 0.5, 0.5, CriticConfig(margin=0.10))[0]
    assert lax is True
    assert strict is False


def test_on_responds_to_iou_threshold():
    """Phone resting on notebook: small overlap should pass when threshold is
    lowered, mirroring the calibration we expect on real data."""
    phone = _b(0.40, 0.45, 0.55, 0.60)
    notebook = _b(0.30, 0.55, 0.70, 0.85)
    permissive = _verify_relation("on", phone, notebook, 0.5, 0.5, CriticConfig(on_iou_threshold=0.01))[0]
    strict = _verify_relation("on", phone, notebook, 0.5, 0.5, CriticConfig(on_iou_threshold=0.50))[0]
    assert permissive is True
    assert strict is False


def test_contains_responds_to_coverage_threshold():
    """Apple half inside bowl: passes only when coverage threshold is relaxed."""
    bowl = _b(0.20, 0.40, 0.80, 0.95)
    apple = _b(0.45, 0.30, 0.65, 0.55)  # straddles top edge of bowl
    relaxed = _verify_relation("contains", bowl, apple, 0.5, 0.5, CriticConfig(contains_coverage_threshold=0.30))[0]
    strict = _verify_relation("contains", bowl, apple, 0.5, 0.5, CriticConfig(contains_coverage_threshold=0.90))[0]
    assert relaxed is True
    assert strict is False


# ── geo_confidence ───────────────────────────────────────────────────────────

def _bc(x1, y1, x2, y2, conf):
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)


def test_geo_confidence_high_for_clear_horizontal_separation():
    cfg = CriticConfig()
    # Objects far apart horizontally, both detected with high confidence.
    a = _bc(0.05, 0.4, 0.20, 0.6, 0.9)
    b = _bc(0.80, 0.4, 0.95, 0.6, 0.9)
    _, ev = _verify_relation("left_of", a, b, 0.5, 0.5, cfg)
    # |dx| ~ 0.75, scale 0.25 -> margin_conf clipped to 1.0; det 0.9.
    assert ev["geo_confidence"] >= 0.85


def test_geo_confidence_low_for_marginal_separation():
    cfg = CriticConfig()
    a = _bc(0.48, 0.4, 0.52, 0.6, 0.9)
    b = _bc(0.50, 0.4, 0.54, 0.6, 0.9)  # |dx| ~ 0.02
    _, ev = _verify_relation("left_of", a, b, 0.5, 0.5, cfg)
    assert ev["geo_confidence"] < 0.2


def test_geo_confidence_scales_with_detection_confidence():
    cfg = CriticConfig()
    far = (0.05, 0.4, 0.20, 0.6)
    near = (0.80, 0.4, 0.95, 0.6)
    _, hi = _verify_relation("left_of", _bc(*far, 0.9), _bc(*near, 0.9), 0.5, 0.5, cfg)
    _, lo = _verify_relation("left_of", _bc(*far, 0.3), _bc(*near, 0.3), 0.5, 0.5, cfg)
    assert hi["geo_confidence"] > lo["geo_confidence"]


def test_depth_relation_confidence_lower_than_position_for_same_detection():
    cfg = CriticConfig()
    # Same detection quality and a modest separation; depth's smaller scale and
    # typically smaller signal should not exceed a clear horizontal call.
    pos_a = _bc(0.10, 0.4, 0.25, 0.6, 0.8)
    pos_b = _bc(0.75, 0.4, 0.90, 0.6, 0.8)
    _, pos = _verify_relation("left_of", pos_a, pos_b, 0.5, 0.5, cfg)
    dep_a = _bc(0.40, 0.4, 0.60, 0.6, 0.8)
    dep_b = _bc(0.40, 0.4, 0.60, 0.6, 0.8)
    _, dep = _verify_relation("behind", dep_a, dep_b, 0.55, 0.50, cfg)  # |dz|=0.05
    assert pos["geo_confidence"] > dep["geo_confidence"]


def test_contains_confidence_capped_to_defer_to_vlm():
    cfg = CriticConfig()
    # Small fully inside big with perfect coverage: without the cap this would
    # earn near-det confidence (~0.9), but contains geometry is non-separable
    # on VSR, so it is capped below the arbitration threshold and defers.
    big = _bc(0.10, 0.10, 0.90, 0.90, 0.9)
    small = _bc(0.40, 0.40, 0.55, 0.55, 0.9)
    _, ev = _verify_relation("contains", big, small, 0.5, 0.5, cfg)
    assert ev["geo_confidence"] <= cfg.contains_confidence_cap
    assert ev["geo_confidence"] < cfg.geo_confidence_arbitration


# ── _compute_crop ────────────────────────────────────────────────────────────

def test_compute_crop_pads_and_clamps_to_unit_square():
    a = _b(0.05, 0.40, 0.20, 0.60)
    b = _b(0.80, 0.40, 0.95, 0.60)
    crop = _compute_crop(a, b, padding=0.10)
    # Bounding box of {a, b} is [0.05, 0.40, 0.95, 0.60]; padding=0.10 pushes
    # both ends past the unit square, so they should clamp.
    assert crop.x1 == 0.0
    assert crop.x2 == 1.0
    assert _close(crop.y1, 0.30)
    assert _close(crop.y2, 0.70)


# ── Script runner so the file is usable without pytest ───────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
