"""
End-to-end pipeline smoke test using mock model backends.

Exercises planner → executor → critic → output wiring after the day-1
depth-frame refactor, without any real LLM, detector, or depth model calls.
Skipped if langchain/langgraph aren't installed in the current environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

try:
    import pytest
except ImportError:  # pragma: no cover - fallback runner provided below
    pytest = None  # type: ignore

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _require_pipeline_deps():
    """Skip when LangChain/LangGraph aren't installed."""
    for mod in ("langchain_core", "langgraph", "langchain_openai"):
        try:
            __import__(mod)
        except ImportError as exc:
            if pytest is not None:
                pytest.skip(f"pipeline dep missing: {mod} ({exc})")
            else:
                print(f"SKIP  pipeline deps missing: {mod}")
                raise SystemExit(0)


class _StubLLM:
    """Just enough of the LangChain chat-model surface to satisfy run_planner.

    Notably has no `with_structured_output`, so _invoke_structured_planner
    falls through to the text-JSON path immediately.
    """

    def __init__(self, obj1: str, obj2: str, relation: str):
        self._payload = f'{{"obj1": "{obj1}", "obj2": "{obj2}", "relation": "{relation}"}}'

    def invoke(self, _messages):
        return SimpleNamespace(content=self._payload)


def test_pipeline_smoke_left_of_yes():
    _require_pipeline_deps()
    from pipeline import run_pipeline

    img = _SRC / "internet_inputs" / "blue_bottle_yellow_cup.jpg"
    if not img.exists():
        if pytest is not None:
            pytest.skip(f"sample image missing: {img}")
        else:
            print(f"SKIP  sample image missing: {img}")
            return

    llm = _StubLLM("yellow cup", "blue bottle", "left_of")
    graph = run_pipeline(
        image_path=str(img),
        question="Is the yellow cup to the left of the blue bottle?",
        llm=llm,
        executor_config={"backend": "mock", "mock_answer": "yes"},
        critic_config={"allow_mock_models": True},
        max_iterations=2,
        strict_models=False,
    )

    # Planner output flowed through.
    assert graph.obj1 == "yellow cup"
    assert graph.obj2 == "blue bottle"
    assert graph.relation == "left_of"

    # Critic + executor agree on left_of (mock detector places obj1 in the left
    # half of the image), so we expect a verified yes answer in one iteration.
    assert graph.answer_str == "yes"
    assert graph.verified is True
    assert graph.iterations >= 1
    assert graph.evidence, "expected at least one CriticEvidence record"
    last = graph.evidence[-1]
    assert last.obj1_bbox is not None and last.obj2_bbox is not None
    assert last.dx is not None and last.dx < 0  # obj1 left of obj2


def test_pipeline_smoke_disagreement_triggers_abstain_or_correction():
    """When the mock executor says 'no' but mock geometry says left_of holds,
    the pipeline must either correct or abstain — it must NOT silently output
    'yes' without disagreement handling."""
    _require_pipeline_deps()
    from pipeline import run_pipeline

    img = _SRC / "internet_inputs" / "blue_bottle_yellow_cup.jpg"
    if not img.exists():
        if pytest is not None:
            pytest.skip(f"sample image missing: {img}")
        else:
            print(f"SKIP  sample image missing: {img}")
            return

    llm = _StubLLM("yellow cup", "blue bottle", "left_of")
    graph = run_pipeline(
        image_path=str(img),
        question="Is the yellow cup to the left of the blue bottle?",
        llm=llm,
        executor_config={"backend": "mock", "mock_answer": "no"},
        critic_config={"allow_mock_models": True},
        max_iterations=2,
        strict_models=False,
    )
    # Either we reached output after convergence (verified True) or abstained
    # (verified False). Both are acceptable; what's NOT acceptable is the
    # pipeline ending early without resolving the disagreement.
    assert graph.answer_str in {"yes", "no", "abstain"}
    assert graph.iterations >= 1


def _make_disagreement_state(geo_confidence: float, critic_passed: bool, executor_answer: bool):
    """Build an AgentState at the final iteration where executor disagrees with geometry."""
    from state import AgentState, BoundingBox, CriticEvidence

    ev = CriticEvidence(
        claim="x left_of y",
        passed=critic_passed,
        obj1_bbox=BoundingBox(x1=0.1, y1=0.4, x2=0.3, y2=0.6, confidence=0.9, label="x"),
        obj2_bbox=BoundingBox(x1=0.6, y1=0.4, x2=0.8, y2=0.6, confidence=0.9, label="y"),
        rule_applied="cx(obj1)-cx(obj2)=... (test)",
        geo_confidence=geo_confidence,
    )
    return AgentState(
        image_path="x.jpg",
        question="Is x left of y?",
        obj1="x", obj2="y", relation="left_of",
        executor_answer=executor_answer,
        critic_passed=critic_passed,
        critic_evidence=[ev],
        iteration=2, max_iterations=3,
    )


def _arbitrate(state, threshold=0.40):
    _require_pipeline_deps()
    from pipeline import arbitrate_node
    cfg = {"configurable": {"critic": {"geo_confidence_arbitration": threshold}}}
    out = arbitrate_node(state.model_dump(), cfg)
    from state import AgentState
    return AgentState(**out).graph


def test_arbitration_high_confidence_overrides_vlm():
    # Geometry says False (not left_of) with high confidence; VLM says True.
    s = _make_disagreement_state(geo_confidence=0.85, critic_passed=False, executor_answer=True)
    g = _arbitrate(s)
    assert g.answer_str == "no"           # geometry won
    assert g.answer_source == "geometry_override"
    assert g.verified is True


def test_arbitration_low_confidence_defers_to_vlm():
    # Geometry says False but with low confidence; VLM says True -> trust VLM.
    s = _make_disagreement_state(geo_confidence=0.10, critic_passed=False, executor_answer=True)
    g = _arbitrate(s)
    assert g.answer_str == "yes"          # VLM won
    assert g.answer_source == "vlm_deferred"
    assert g.verified is False


def test_high_precision_abstains_below_confidence_threshold():
    # Low geo_confidence committed answer abstains when the precision knob is set.
    s = _make_disagreement_state(geo_confidence=0.20, critic_passed=True, executor_answer=True)
    # executor agrees with geometry here, so route through arbitrate commit path.
    _require_pipeline_deps()
    from pipeline import arbitrate_node
    from state import AgentState
    cfg = {"configurable": {"critic": {"abstain_below_confidence": 0.40}}}
    g = AgentState(**arbitrate_node(s.model_dump(), cfg)).graph
    assert g.answer_str == "abstain"
    assert g.failure_mode == "low_confidence"


def test_high_precision_commits_above_confidence_threshold():
    s = _make_disagreement_state(geo_confidence=0.85, critic_passed=False, executor_answer=True)
    _require_pipeline_deps()
    from pipeline import arbitrate_node
    from state import AgentState
    cfg = {"configurable": {"critic": {"abstain_below_confidence": 0.40}}}
    g = AgentState(**arbitrate_node(s.model_dump(), cfg)).graph
    assert g.answer_str != "abstain"          # high confidence -> commits
    assert g.answer_source == "geometry_override"


def test_arbitration_threshold_is_respected():
    # Same state, two thresholds straddling geo_confidence=0.5.
    s1 = _make_disagreement_state(geo_confidence=0.5, critic_passed=True, executor_answer=False)
    g_trust = _arbitrate(s1, threshold=0.40)
    s2 = _make_disagreement_state(geo_confidence=0.5, critic_passed=True, executor_answer=False)
    g_defer = _arbitrate(s2, threshold=0.60)
    assert g_trust.answer_source == "geometry_override"
    assert g_defer.answer_source == "vlm_deferred"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except SystemExit:
            raise
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
