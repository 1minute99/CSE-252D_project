"""
Streamlit demo for the Spatial Evidence Agent.

Lets users upload an image, ask a binary spatial question, and inspect the
Critic's per-iteration geometric evidence (bboxes, depths, applied rule).
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_openai import ChatOpenAI

from config import CriticConfig
from env_loader import get_openai_api_key, load_project_env
from pipeline import run_pipeline
from visualize import annotate_image

# --- Page Config ---------------------------------------------------------------
st.set_page_config(page_title="Spatial Evidence Agent", page_icon="👁️", layout="wide")
st.title("Spatial Evidence Agent")
st.caption(
    "Upload an RGB image and ask a binary spatial question. The Planner parses "
    "the question, the Executor proposes a yes/no, and the Critic verifies the "
    "claim with Grounding-DINO + Depth Anything V2."
)

load_project_env()

# --- Backend descriptors -------------------------------------------------------
# Two backends, but only one of them is "really" different — the Critic always
# runs Grounding-DINO + Depth Anything V2 locally on GPU. The choice below only
# affects the Planner LLM and the Executor VLM.
BACKEND_OPENAI = "OpenAI API"
BACKEND_LOCAL_OPENAI_COMPAT = "Local OpenAI-compatible server (Ollama / vLLM / LM Studio)"

# Persist API key across re-runs so users don't retype.
if "openai_key" not in st.session_state:
    st.session_state.openai_key = get_openai_api_key()

# --- Sidebar -------------------------------------------------------------------
st.sidebar.header("Settings")
default_critic_cfg = CriticConfig()
backend_choice = st.sidebar.radio(
    "Backend (Planner + Executor)",
    [BACKEND_OPENAI, BACKEND_LOCAL_OPENAI_COMPAT],
    help=(
        "The Critic always uses local Grounding-DINO + Depth Anything V2. "
        "This setting only controls which LLM/VLM serves the Planner and "
        "Executor."
    ),
)

if backend_choice == BACKEND_OPENAI:
    st.session_state.openai_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_key,
        help="Read from .env at startup if present.",
    )
    api_key = st.session_state.openai_key
    base_url = None
    planner_model = st.sidebar.text_input("Planner model", value="gpt-4o-mini")
    vision_model = st.sidebar.text_input("Executor (vision) model", value="gpt-4o")
else:
    st.sidebar.info(
        "Point this at any OpenAI-compatible local server. Defaults assume "
        "Ollama at localhost:11434."
    )
    base_url = st.sidebar.text_input("Base URL", value="http://localhost:11434/v1")
    api_key = st.sidebar.text_input("API key (any non-empty value)", value="ollama")
    planner_model = st.sidebar.text_input("Planner model", value="llava")
    vision_model = st.sidebar.text_input("Executor (vision) model", value="llava")

max_iterations = st.sidebar.slider("Max correction iterations (k)", 1, 3, 3)

with st.sidebar.expander("Advanced critic thresholds", expanded=False):
    margin = st.slider("Position/depth margin", 0.0, 0.20, default_critic_cfg.margin, step=0.01)
    on_iou = st.slider("'on' IoU threshold", 0.0, 0.50, default_critic_cfg.on_iou_threshold, step=0.01)
    contains_cov = st.slider(
        "'contains' coverage threshold",
        0.0,
        1.0,
        default_critic_cfg.contains_coverage_threshold,
        step=0.05,
    )
    area_ratio = st.slider(
        "'contains' area ratio threshold",
        0.0,
        1.0,
        default_critic_cfg.area_ratio_threshold,
        step=0.05,
    )
    crop_padding = st.slider(
        "Active-perception crop padding",
        0.0,
        0.30,
        default_critic_cfg.crop_padding,
        step=0.01,
    )

# --- Main pane -----------------------------------------------------------------
uploaded_file = st.file_uploader("Image (JPG / PNG)", type=["jpg", "jpeg", "png"])
question = st.text_input(
    "Spatial question",
    placeholder="e.g., Is the yellow cup to the right of the blue bottle?",
)

run_clicked = st.button("Analyze", type="primary")

if run_clicked:
    # --- Input validation -----------------------------------------------------
    if not uploaded_file:
        st.error("Upload an image first.")
        st.stop()
    if not question.strip():
        st.error("Type a spatial question.")
        st.stop()
    if backend_choice == BACKEND_OPENAI and not api_key:
        st.error("OpenAI backend requires an API key in the sidebar.")
        st.stop()

    # --- Persist upload to a temp path the pipeline can read ------------------
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_image_path = tmp_file.name
    annotated_img_path = None

    try:
        with st.status("Running Spatial Evidence Agent", expanded=True) as status:
            status.write("Building LLM client…")
            llm = ChatOpenAI(
                model=planner_model,
                api_key=api_key or "missing",
                base_url=base_url,
                temperature=0,
                max_retries=0,
                timeout=60,
            )

            exec_cfg = {
                "backend": "openai",  # the local case is also OpenAI-API-shaped
                "openai_key": api_key,
                "openai_base": base_url,
                "model": vision_model,
            }
            critic_cfg = {
                "margin": margin,
                "on_iou_threshold": on_iou,
                "contains_coverage_threshold": contains_cov,
                "area_ratio_threshold": area_ratio,
                "crop_padding": crop_padding,
                "allow_mock_models": False,
            }

            status.write("Pipeline running (planner → executor → critic)…")
            graph = run_pipeline(
                image_path=temp_image_path,
                question=question,
                llm=llm,
                executor_config=exec_cfg,
                critic_config=critic_cfg,
                max_iterations=max_iterations,
                strict_models=True,
            )

            status.write("Rendering annotated image…")
            annotated_img_path = temp_image_path + "_annotated.jpg"
            annotate_image(temp_image_path, graph, out_path=annotated_img_path)
            status.update(label="Done", state="complete")

        # --- Results ----------------------------------------------------------
        st.divider()
        verdict = graph.answer_str.upper()
        if graph.verified:
            st.success(f"Verified answer: **{verdict}** (confidence {graph.confidence:.2f})")
        elif graph.answer_str == "abstain":
            st.warning(f"Abstained — failure mode: `{graph.failure_mode}`")
        else:
            st.info(f"Answer: **{verdict}**")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Annotated image")
            st.image(annotated_img_path, use_container_width=True)
        with col2:
            st.subheader("Predicate")
            st.markdown(
                f"- **obj1**: `{graph.obj1}`\n"
                f"- **relation**: `{graph.relation}`\n"
                f"- **obj2**: `{graph.obj2}`\n"
                f"- **iterations**: {graph.iterations}\n"
                f"- **verified**: {graph.verified}\n"
                f"- **failure_mode**: `{graph.failure_mode or '—'}`"
            )

        st.subheader("Per-iteration evidence")
        if not graph.evidence:
            st.write("(no Critic evidence recorded)")
        for i, ev in enumerate(graph.evidence, 1):
            tag = "PASS" if ev.passed else "FAIL"
            with st.expander(f"Iteration {i} — {tag}", expanded=(i == len(graph.evidence))):
                st.code(ev.rule_applied or "(no rule applied)", language="text")
                if ev.failure_reason:
                    st.write(f"**Failure reason:** `{ev.failure_reason}`")
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**obj1 detection**")
                    if ev.obj1_bbox is not None:
                        b = ev.obj1_bbox
                        st.write(
                            f"label `{b.label}`, conf {b.confidence:.2f}, "
                            f"box [{b.x1:.2f}, {b.y1:.2f}, {b.x2:.2f}, {b.y2:.2f}]"
                        )
                        if ev.obj1_depth is not None:
                            st.write(f"depth = {ev.obj1_depth:.3f}")
                    else:
                        st.write("(not detected)")
                with cols[1]:
                    st.markdown("**obj2 detection**")
                    if ev.obj2_bbox is not None:
                        b = ev.obj2_bbox
                        st.write(
                            f"label `{b.label}`, conf {b.confidence:.2f}, "
                            f"box [{b.x1:.2f}, {b.y1:.2f}, {b.x2:.2f}, {b.y2:.2f}]"
                        )
                        if ev.obj2_depth is not None:
                            st.write(f"depth = {ev.obj2_depth:.3f}")
                    else:
                        st.write("(not detected)")

        if graph.crop_history:
            st.subheader("Active-perception crops")
            for i, crop in enumerate(graph.crop_history, 1):
                st.write(
                    f"{i}. `[{crop.x1:.2f}, {crop.y1:.2f}, {crop.x2:.2f}, {crop.y2:.2f}]` — {crop.reason}"
                )

        with st.expander("Raw Spatial Evidence Graph (JSON)", expanded=False):
            st.json(graph.model_dump())

    except Exception as exc:
        # Surface where the failure came from when the pipeline raises a strict error.
        message = str(exc)
        st.error(f"Pipeline error: {message}")
        if "Planner" in message:
            st.caption("Hint: the Planner couldn't parse the question. Try simpler phrasing.")
        elif "Executor" in message:
            st.caption("Hint: check the backend URL / API key / model name in the sidebar.")
        elif "checkpoints" in message.lower() or "groundingdino" in message.lower():
            st.caption(
                "Hint: place GroundingDINO_SwinT_OGC.py + groundingdino_swint_ogc.pth in src/checkpoints/."
            )
        with st.expander("Full traceback", expanded=False):
            import traceback
            st.code(traceback.format_exc(), language="text")

    finally:
        for path in (temp_image_path, annotated_img_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
