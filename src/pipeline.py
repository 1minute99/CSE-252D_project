"""
LangGraph orchestrator.

Wires Planner -> Executor -> Critic with a bounded active-perception loop:

START -> planner -> executor -> critic -> output
                              -> correction -> executor
                              -> abstain
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

import critic
import depth
import executor
import planner
from config import CriticConfig
from state import AgentState, CriticEvidence, SpatialEvidenceGraph

logger = logging.getLogger(__name__)


def planner_node(state: dict, config: RunnableConfig) -> dict:
    s = AgentState(**state)
    llm = config["configurable"]["llm"]
    strict = bool(config["configurable"].get("strict_models", False))
    s = planner.run_planner(s, llm, strict=strict)
    return s.model_dump()


def executor_node(state: dict, config: RunnableConfig) -> dict:
    s = AgentState(**state)
    exec_cfg = config["configurable"].get("executor", {})
    if config["configurable"].get("strict_models", False):
        exec_cfg = {**exec_cfg, "strict": True}
    s = executor.run_executor(s, exec_cfg)
    return s.model_dump()


def critic_node(state: dict, config: RunnableConfig) -> dict:
    s = AgentState(**state)
    critic_cfg = config["configurable"].get("critic", {})
    s = critic.run_critic(s, critic_cfg)
    return s.model_dump()


def correction_node(state: dict, config: RunnableConfig) -> dict:
    """
    Increment iteration counter; the next executor call will use
    state.current_crop, already computed by the Critic.
    """
    s = AgentState(**state)
    s.iteration += 1
    logger.info(f"[Correction] Advancing to iteration {s.iteration}")
    return s.model_dump()


def _latest_evidence(s: AgentState) -> CriticEvidence | None:
    return s.critic_evidence[-1] if s.critic_evidence else None


def _has_geometric_evidence(s: AgentState) -> bool:
    ev = _latest_evidence(s)
    if ev is None:
        return False
    if ev.obj1_bbox is None or ev.obj2_bbox is None:
        return False
    if ev.rule_applied.startswith("unknown relation"):
        return False
    return bool(ev.rule_applied)


def _executor_agrees_with_geometry(s: AgentState) -> bool:
    if not _has_geometric_evidence(s):
        return False
    if s.executor_answer is None:
        return True
    return bool(s.executor_answer) == bool(s.critic_passed)


def _below_confidence(s: AgentState, cfg: CriticConfig) -> bool:
    """High-precision mode: should this committed answer abstain for low
    geometric confidence?"""
    if cfg.abstain_below_confidence <= 0.0:
        return False
    ev = _latest_evidence(s)
    geo_conf = float(ev.geo_confidence) if ev is not None else 0.0
    return geo_conf < cfg.abstain_below_confidence


def _low_confidence_abstain(s: AgentState) -> dict:
    graph = SpatialEvidenceGraph(
        question=s.question, obj1=s.obj1, obj2=s.obj2, relation=s.relation,
        answer=None, answer_str="abstain", confidence=0.0,
        evidence=s.critic_evidence, iterations=s.iteration + 1,
        crop_history=s.crop_history, verified=False,
        failure_mode="low_confidence", answer_source="abstain_low_conf",
    )
    s.graph = _attach_signals(graph, s)
    s.done = True
    s.abstain = True
    logger.info("[Abstain] low_confidence (below abstain_below_confidence)")
    return s.model_dump()


def _attach_signals(graph: SpatialEvidenceGraph, s: AgentState) -> SpatialEvidenceGraph:
    """Record per-item confidence signals on the SEG for offline risk analysis."""
    ev = _latest_evidence(s)
    graph.executor_confidence = float(s.executor_confidence)
    graph.geo_confidence = float(ev.geo_confidence) if ev is not None else 0.0
    if s.executor_answer is None or not _has_geometric_evidence(s):
        graph.executor_agreed = None
    else:
        graph.executor_agreed = bool(s.executor_answer) == bool(s.critic_passed)
    return graph


def output_node(state: dict, config: RunnableConfig) -> dict:
    """Build the final Spatial Evidence Graph from geometric evidence."""
    s = AgentState(**state)
    cfg = CriticConfig.from_mapping(config["configurable"].get("critic", {}))

    if _below_confidence(s, cfg):
        return _low_confidence_abstain(s)

    answer = bool(s.critic_passed)
    answer_str = "yes" if answer else "no"
    geometry_only = s.executor_answer is None

    graph = SpatialEvidenceGraph(
        question=s.question,
        obj1=s.obj1,
        obj2=s.obj2,
        relation=s.relation,
        answer=answer,
        answer_str=answer_str,
        confidence=0.7 if geometry_only else 0.9,
        evidence=s.critic_evidence,
        iterations=s.iteration + 1,
        crop_history=s.crop_history,
        verified=True,
        failure_mode="",
        answer_source="geometry_only" if geometry_only else "agreement",
    )
    s.graph = _attach_signals(graph, s)
    s.done = True
    logger.info(f"[Output] Final answer: {answer_str} (verified=True)")
    return s.model_dump()


def arbitrate_node(state: dict, config: RunnableConfig) -> dict:
    """
    Final-iteration disagreement between Executor and geometry.

    Rather than always abstaining (which we found discards many correct
    answers), decide by geometric confidence:
      - high geo_confidence  -> commit the geometry answer (override the VLM)
      - low  geo_confidence  -> defer to the VLM's answer
    Abstain only when there is no usable geometric evidence at all
    (e.g. detector miss) and no VLM answer to fall back on.
    """
    s = AgentState(**state)
    cfg = CriticConfig.from_mapping(config["configurable"].get("critic", {}))
    ev = _latest_evidence(s)

    # No geometry to arbitrate with (detector miss).
    if not _has_geometric_evidence(s):
        # Full-coverage mode: defer to the VLM instead of abstaining.
        if cfg.vlm_fallback_on_miss and s.executor_answer is not None:
            answer = bool(s.executor_answer)
            answer_str = "yes" if answer else "no"
            graph = SpatialEvidenceGraph(
                question=s.question, obj1=s.obj1, obj2=s.obj2, relation=s.relation,
                answer=answer, answer_str=answer_str, confidence=0.5,
                evidence=s.critic_evidence, iterations=s.iteration + 1,
                crop_history=s.crop_history, verified=False,
                failure_mode="", answer_source="vlm_fallback",
            )
            s.graph = _attach_signals(graph, s)
            s.done = True
            logger.info(f"[Arbitrate] vlm_fallback (detector miss) -> {answer_str}")
            return s.model_dump()
        return abstain_node(state, config)

    geo_conf = float(ev.geo_confidence) if ev is not None else 0.0
    if _below_confidence(s, cfg):
        return _low_confidence_abstain(s)
    trust_geometry = geo_conf >= cfg.geo_confidence_arbitration

    if trust_geometry:
        answer = bool(s.critic_passed)
        source = "geometry_override"
        confidence = round(0.6 + 0.3 * geo_conf, 3)
    elif s.executor_answer is not None:
        answer = bool(s.executor_answer)
        source = "vlm_deferred"
        confidence = 0.55
    else:
        # Geometry weak and no VLM answer -> abstain.
        return abstain_node(state, config)

    answer_str = "yes" if answer else "no"
    graph = SpatialEvidenceGraph(
        question=s.question,
        obj1=s.obj1,
        obj2=s.obj2,
        relation=s.relation,
        answer=answer,
        answer_str=answer_str,
        confidence=confidence,
        evidence=s.critic_evidence,
        iterations=s.iteration + 1,
        crop_history=s.crop_history,
        verified=trust_geometry,
        failure_mode="",
        answer_source=source,
    )
    s.graph = _attach_signals(graph, s)
    s.done = True
    logger.info(
        f"[Arbitrate] {source} -> {answer_str} (geo_confidence={geo_conf:.3f}, "
        f"threshold={cfg.geo_confidence_arbitration})"
    )
    return s.model_dump()


def abstain_node(state: dict, config: RunnableConfig) -> dict:
    """Called when parsing fails or max correction iterations are reached."""
    s = AgentState(**state)

    failure = "unknown"
    if s.error.startswith("Planner error"):
        failure = "planner_parse_error"
    elif s.critic_evidence:
        ev = s.critic_evidence[-1]
        if ev.failure_reason.startswith("detector_miss"):
            failure = "detector_miss"
        elif "depth" in ev.failure_reason:
            failure = "depth_noise"
        elif _has_geometric_evidence(s):
            failure = "vlm_bias"

    graph = SpatialEvidenceGraph(
        question=s.question,
        obj1=s.obj1,
        obj2=s.obj2,
        relation=s.relation,
        answer=None,
        answer_str="abstain",
        confidence=0.0,
        evidence=s.critic_evidence,
        iterations=s.iteration + 1,
        crop_history=s.crop_history,
        verified=False,
        failure_mode=failure,
    )
    s.graph = _attach_signals(graph, s)
    s.done = True
    s.abstain = True
    logger.warning(f"[Abstain] {failure}")
    return s.model_dump()


def route_after_planner(state: dict) -> str:
    s = AgentState(**state)
    if s.error.startswith("Planner error") and not (s.obj1 and s.obj2 and s.relation):
        return "abstain"
    return "executor"


def route_after_critic(state: dict) -> str:
    s = AgentState(**state)
    if _executor_agrees_with_geometry(s):
        return "output"
    if s.iteration >= s.max_iterations - 1:
        # Out of correction budget and still disagreeing: arbitrate by
        # geometric confidence instead of unconditionally abstaining.
        return "arbitrate"
    return "correction"


def build_graph() -> StateGraph:
    g = StateGraph(dict)

    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("critic", critic_node)
    g.add_node("correction", correction_node)
    g.add_node("output", output_node)
    g.add_node("arbitrate", arbitrate_node)
    g.add_node("abstain", abstain_node)

    g.set_entry_point("planner")
    g.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "executor": "executor",
            "abstain": "abstain",
        },
    )
    g.add_edge("executor", "critic")
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "output": "output",
            "correction": "correction",
            "arbitrate": "arbitrate",
        },
    )

    g.add_edge("correction", "executor")
    g.add_edge("output", END)
    g.add_edge("arbitrate", END)
    g.add_edge("abstain", END)

    return g.compile()


def run_pipeline(
    image_path: str,
    question: str,
    llm: Any,
    executor_config: dict | None = None,
    critic_config: dict | None = None,
    max_iterations: int = 3,
    strict_models: bool = False,
) -> SpatialEvidenceGraph:
    """
    High-level entry point.

    Args:
        image_path: Path to the RGB image.
        question: Binary spatial question in English.
        llm: LangChain ChatModel for the Planner.
        executor_config: dict with executor backend settings.
        critic_config: dict with detector/depth fallback settings.
        max_iterations: Max executor/critic attempts, in {1, 2, 3}.
        strict_models: If true, Planner/Executor model failures raise instead
            of falling back to parser regex or geometry-only behavior.
    """
    pipeline = build_graph()

    init_state = AgentState(
        image_path=str(image_path),
        question=question,
        max_iterations=max_iterations,
    ).model_dump()

    config = {
        "configurable": {
            "llm": llm,
            "executor": executor_config or {"backend": "local"},
            "critic": critic_config or {},
            "strict_models": strict_models,
        }
    }

    try:
        result = pipeline.invoke(init_state, config=config)
    finally:
        # Release the (single-entry) cached depth map so memory stays flat
        # when run_pipeline is called many times in evaluate.py.
        depth.clear_depth_cache()
    final = AgentState(**result)
    return final.graph
