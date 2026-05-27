"""
Centralized configuration for the Critic's geometry rules.

Thresholds were previously module constants in critic.py. Pulling them into a
dataclass lets ablation/calibration scripts vary them per run without editing
source.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping


@dataclass
class CriticConfig:
    # Position/depth slack in normalised image coords. 0.02 was too tight: on
    # VSR-200 it fired left/above verdicts on near-aligned objects (FP=17 on
    # 73 position items). The FP curve is monotone in the margin and the whole
    # 0.02->0.10 range loses only ~2 TPs, so 0.08 (FP=12) commits a verdict
    # only when objects are meaningfully separated without grabbing the
    # endpoint.
    margin: float = 0.08
    on_iou_threshold: float = 0.05
    contains_coverage_threshold: float = 0.70
    area_ratio_threshold: float = 0.70
    crop_padding: float = 0.05
    # When the Executor disagrees with geometry at the final iteration, commit
    # the geometry answer if geo_confidence >= this threshold, otherwise defer
    # to the Executor (VLM). Set to 1.0 to recover the old abstain-on-disagree
    # behavior.
    geo_confidence_arbitration: float = 0.40
    # Monocular depth is too unreliable to override the VLM: on VSR-200,
    # high-confidence depth overrides were only ~15% correct. Cap behind/
    # in_front geo_confidence below the arbitration threshold so depth
    # relations always defer to the VLM on disagreement. Set to 1.0 to allow
    # depth overrides.
    depth_confidence_cap: float = 0.30
    # Containment geometry is non-separable on VSR: most true-contains items
    # have obj2 larger than obj1 with coverage~0 (captions whose boxes don't
    # reflect visual nesting), so no coverage/area threshold beats 3/12 TP.
    # Cap contains geo_confidence below the arbitration threshold so it defers
    # to the VLM instead of overruling it. Set to 1.0 to allow contains
    # overrides.
    contains_confidence_cap: float = 0.30
    # When True, a detector miss (no geometric evidence) falls back to the
    # VLM's answer instead of abstaining — i.e. full-coverage mode. When False
    # (default), the system abstains, preserving the selective-prediction
    # property that every committed answer carries geometric evidence.
    vlm_fallback_on_miss: bool = False
    # High-precision / low-coverage mode: abstain on any committed answer whose
    # geometric confidence is below this threshold. 0.0 disables it (default).
    # On VSR-200, ~0.40 yields ~0.85 accuracy at ~0.34 coverage.
    abstain_below_confidence: float = 0.0
    allow_mock_models: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | "CriticConfig" | None) -> "CriticConfig":
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in dict(data).items() if k in known}
        return cls(**kwargs)
