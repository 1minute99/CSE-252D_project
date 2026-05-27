"""
Critic Agent - geometric rule engine.

Verifies the parsed spatial relation using open-vocabulary detections,
relative monocular depth, and deterministic geometry rules.
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

import depth
import detector
from config import CriticConfig
from state import BoundingBox, CriticEvidence, CropRegion

logger = logging.getLogger(__name__)


def _make_bbox(det: dict) -> BoundingBox:
    b = det["bbox"]
    return BoundingBox(
        x1=float(b[0]),
        y1=float(b[1]),
        x2=float(b[2]),
        y2=float(b[3]),
        confidence=float(det.get("confidence", 0.0)),
        label=str(det.get("label", "")),
    )


def _map_bbox_to_original(box: BoundingBox, crop: CropRegion | None) -> BoundingBox:
    if crop is None:
        return box
    width = crop.x2 - crop.x1
    height = crop.y2 - crop.y1
    return BoundingBox(
        x1=crop.x1 + box.x1 * width,
        y1=crop.y1 + box.y1 * height,
        x2=crop.x1 + box.x2 * width,
        y2=crop.y1 + box.y2 * height,
        confidence=box.confidence,
        label=box.label,
    )


def _intersection_area(b1: BoundingBox, b2: BoundingBox) -> float:
    ix1 = max(b1.x1, b2.x1)
    iy1 = max(b1.y1, b2.y1)
    ix2 = min(b1.x2, b2.x2)
    iy2 = min(b1.y2, b2.y2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _verify_relation(
    relation: str,
    b1: BoundingBox,
    b2: BoundingBox,
    d1: float,
    d2: float,
    cfg: CriticConfig,
) -> tuple[bool, dict]:
    dx = b1.cx - b2.cx
    dy = b1.cy - b2.cy
    dz = d1 - d2
    iou = b1.iou(b2)

    margin = cfg.margin
    on_iou = cfg.on_iou_threshold
    cov_thr = cfg.contains_coverage_threshold
    area_thr = cfg.area_ratio_threshold

    evidence = {
        "dx": round(dx, 4),
        "dy": round(dy, 4),
        "dz": round(dz, 4),
        "iou": round(iou, 4),
    }

    # Geometric confidence = detection quality × how decisively the measurement
    # clears zero (signal strength). Depth relations use |dz|, which is smaller
    # and noisier, so they naturally earn lower confidence — letting the
    # arbitration step defer those to the VLM, which is stronger on depth.
    det_conf = min(b1.confidence, b2.confidence)
    if relation in ("left_of", "right_of"):
        signal, scale = abs(dx), 0.25
    elif relation in ("above", "below"):
        signal, scale = abs(dy), 0.25
    elif relation in ("behind", "in_front"):
        signal, scale = abs(dz), 0.20
    elif relation == "on":
        signal, scale = abs(dy), 0.20
    elif relation == "contains":
        inter_c = _intersection_area(b1, b2)
        signal, scale = inter_c / max(b2.area, 1e-6), 1.0
    else:
        signal, scale = 0.0, 1.0
    margin_conf = max(0.0, min(1.0, signal / scale))
    geo_confidence = det_conf * margin_conf
    if relation in ("behind", "in_front"):
        # Monocular depth is too unreliable to override the VLM (see
        # CriticConfig.depth_confidence_cap); keep it below the arbitration
        # threshold so depth relations defer rather than override.
        geo_confidence = min(geo_confidence, cfg.depth_confidence_cap)
    evidence["geo_confidence"] = round(geo_confidence, 4)

    if relation == "left_of":
        passed = dx < -margin
        evidence["rule_applied"] = f"cx(obj1)-cx(obj2)={dx:.3f} < -{margin}"
    elif relation == "right_of":
        passed = dx > margin
        evidence["rule_applied"] = f"cx(obj1)-cx(obj2)={dx:.3f} > {margin}"
    elif relation == "above":
        passed = dy < -margin
        evidence["rule_applied"] = f"cy(obj1)-cy(obj2)={dy:.3f} < -{margin}"
    elif relation == "below":
        passed = dy > margin
        evidence["rule_applied"] = f"cy(obj1)-cy(obj2)={dy:.3f} > {margin}"
    elif relation == "behind":
        passed = dz > margin
        evidence["rule_applied"] = f"depth(obj1)-depth(obj2)={dz:.3f} > {margin}"
    elif relation == "in_front":
        passed = dz < -margin
        evidence["rule_applied"] = f"depth(obj1)-depth(obj2)={dz:.3f} < -{margin}"
    elif relation == "on":
        passed = (dy < -margin) and (iou > on_iou)
        evidence["rule_applied"] = (
            f"cy diff={dy:.3f}<-{margin} AND IoU={iou:.3f}>{on_iou}"
        )
    elif relation == "contains":
        inter = _intersection_area(b1, b2)
        coverage = inter / max(b2.area, 1e-6)
        area_ratio = b2.area / max(b1.area, 1e-6)
        area_ok = area_ratio < area_thr
        passed = coverage > cov_thr and area_ok
        evidence["rule_applied"] = (
            f"coverage(obj2 in obj1)={coverage:.3f}>{cov_thr} "
            f"AND area(b2)/area(b1)={area_ratio:.2f}<{area_thr}"
        )
    else:
        passed = False
        evidence["rule_applied"] = f"unknown relation: {relation}"

    return passed, evidence


def _compute_crop(b1: BoundingBox, b2: BoundingBox, padding: float) -> CropRegion:
    x1 = max(0.0, min(b1.x1, b2.x1) - padding)
    y1 = max(0.0, min(b1.y1, b2.y1) - padding)
    x2 = min(1.0, max(b1.x2, b2.x2) + padding)
    y2 = min(1.0, max(b1.y2, b2.y2) + padding)
    return CropRegion(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        reason=f"Disputed region around {b1.label!r} and {b2.label!r}",
    )


def _load_image_with_crop(image_path: str, crop: CropRegion | None) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    if crop is None:
        return img
    w, h = img.size
    return img.crop(
        (
            int(crop.x1 * w),
            int(crop.y1 * h),
            int(crop.x2 * w),
            int(crop.y2 * h),
        )
    )


def _append_evidence(state: Any, evidence: CriticEvidence) -> None:
    state.critic_evidence.append(evidence)


def run_critic(state: Any, config: dict | CriticConfig | None = None) -> Any:
    """
    Critic node.

    Accepts a CriticConfig or a plain dict (forwards-compatible with the
    earlier {"allow_mock_models": bool} shape).
    """
    cfg = CriticConfig.from_mapping(config)
    crop = state.current_crop
    img = _load_image_with_crop(state.image_path, crop)

    obj1_query = state.executor_claims[0] if state.executor_claims else state.obj1
    obj2_query = state.executor_claims[1] if len(state.executor_claims) > 1 else state.obj2
    prompt = f"{obj1_query} . {obj2_query}"

    logger.info(f"[Critic] iter={state.iteration} | query: {prompt!r}")

    evidence = CriticEvidence(
        claim=f"{state.obj1} {state.relation.replace('_', ' ')} {state.obj2}"
    )

    try:
        detections = detector.detect(img, prompt, allow_mock=cfg.allow_mock_models)
        det1 = detector.best_match(detections, obj1_query)
        det2 = detector.best_match(detections, obj2_query)
    except Exception as exc:
        logger.error(f"[Critic] Detector error: {exc}")
        evidence.passed = False
        evidence.failure_reason = f"detector_miss: {exc}"
        state.critic_passed = False
        _append_evidence(state, evidence)
        return state

    if det1 is None or det2 is None:
        missing = obj1_query if det1 is None else obj2_query
        logger.warning(f"[Critic] Detector missed: {missing!r}")
        evidence.passed = False
        evidence.failure_reason = f"detector_miss: {missing!r} not found"
        state.critic_passed = False
        _append_evidence(state, evidence)

        if state.iteration < state.max_iterations - 1:
            new_crop = CropRegion(
                x1=0.0,
                y1=0.0,
                x2=1.0,
                y2=1.0,
                reason=f"Full-image re-examine after detector miss on {missing!r}",
            )
            state.current_crop = new_crop
            state.crop_history.append(new_crop)
        return state

    local_b1 = _make_bbox(det1)
    local_b2 = _make_bbox(det2)
    b1 = _map_bbox_to_original(local_b1, crop)
    b2 = _map_bbox_to_original(local_b2, crop)
    evidence.obj1_bbox = b1
    evidence.obj2_bbox = b2

    try:
        # Depth must come from the ORIGINAL frame so values from successive
        # active-perception crops are comparable. estimate_depth_for_path is
        # cached on image_path so iterations 2/3 don't recompute.
        depth_map = depth.estimate_depth_for_path(
            state.image_path, allow_mock=cfg.allow_mock_models
        )
        b1_box = [b1.x1, b1.y1, b1.x2, b1.y2]
        b2_box = [b2.x1, b2.y1, b2.x2, b2.y2]
        d1 = depth.median_depth_in_box(depth_map, b1_box)
        d2 = depth.median_depth_in_box(depth_map, b2_box)
    except Exception as exc:
        logger.error(f"[Critic] Depth error: {exc}")
        evidence.passed = False
        evidence.failure_reason = f"depth_noise: {exc}"
        state.critic_passed = False
        _append_evidence(state, evidence)
        return state

    evidence.obj1_depth = round(d1, 4)
    evidence.obj2_depth = round(d2, 4)

    passed, geo_ev = _verify_relation(state.relation, b1, b2, d1, d2, cfg)
    evidence.passed = passed
    evidence.dx = geo_ev.get("dx")
    evidence.dy = geo_ev.get("dy")
    evidence.dz = geo_ev.get("dz")
    evidence.iou = geo_ev.get("iou")
    evidence.geo_confidence = geo_ev.get("geo_confidence", 0.0)
    evidence.rule_applied = geo_ev.get("rule_applied", "")

    if not passed:
        evidence.failure_reason = f"relation_false: {evidence.rule_applied}"

    state.critic_passed = passed
    _append_evidence(state, evidence)

    logger.info(
        f"[Critic] relation_true={passed} | rule={evidence.rule_applied} | "
        f"b1=({b1.cx:.2f},{b1.cy:.2f}) b2=({b2.cx:.2f},{b2.cy:.2f}) "
        f"d1={d1:.3f} d2={d2:.3f}"
    )

    if not _executor_answer_matches(state) and state.iteration < state.max_iterations - 1:
        new_crop = _compute_crop(b1, b2, cfg.crop_padding)
        state.crop_history.append(new_crop)
        state.current_crop = new_crop
        logger.info(f"[Critic] Scheduled crop: {new_crop}")

    return state


def _executor_answer_matches(state: Any) -> bool:
    if state.executor_answer is None:
        return True
    return bool(state.executor_answer) == bool(state.critic_passed)
