"""Streamlit frontend for the Autodesk Agentic RAG app."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")

import streamlit as st
import streamlit.components.v1 as components


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent import AutodeskRAGAgent, NO_ANSWER
from src.config import COLLECTION_OPTIONS, HYBRID_BACKEND_NAME, OPTION_3_LABEL, get_settings
from src.retriever import bm25_index_exists, vectorstore_exists


def _patch_torch_classes_for_streamlit_watcher() -> None:
    """Prevent Streamlit's autoreload watcher from tripping on torch.classes.

    PyTorch exposes ``torch.classes.__path__`` as a dynamic custom-class proxy.
    Streamlit's source watcher expects module paths to be ordinary iterables,
    so touching that proxy can print a noisy "Tried to instantiate class
    '__path__._path'" warning. Replacing only this path attribute keeps
    Streamlit autoreload enabled while avoiding the false alarm.
    """

    try:
        import torch

        torch.classes.__path__ = []
    except Exception:
        pass


_patch_torch_classes_for_streamlit_watcher()


def _print_startup_help() -> None:
    """Print terminal guidance once when the Streamlit process starts."""

    if os.environ.get("AUTODESK_RAG_STARTUP_HELP_PRINTED") == "1":
        return
    os.environ["AUTODESK_RAG_STARTUP_HELP_PRINTED"] = "1"

    green = "\033[92m"
    cyan = "\033[96m"
    yellow = "\033[93m"
    red = "\033[91m"
    bold = "\033[1m"
    reset = "\033[0m"

    print(
        "\n"
        f"{bold}{green}Autodesk Agentic RAG is starting.{reset}\n"
        f"{cyan}Open the app:{reset} {bold}http://localhost:8502{reset}\n"
        f"{yellow}To close the app:{reset} press {bold}Ctrl+C{reset} in this terminal.\n"
        f"{red}If the port stays busy:{reset} run "
        f"{bold}Get-NetTCPConnection -LocalPort 8502 | Select-Object -Expand OwningProcess | Stop-Process -Force{reset}\n"
    )


_print_startup_help()

st.set_page_config(page_title="Autodesk Agentic RAG", page_icon="ADSK", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stForm"] {border:1px solid rgba(17,24,39,.12); border-radius:.5rem; padding:1rem; background:#fff;}
    div[data-testid="stTextInput"] input {font-size:1.02rem; min-height:3.1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Autodesk Agentic RAG")
st.caption("Hybrid BM25 keyword plus Chroma vector retrieval over an Autodesk corpus, with strict evidence gating and web fallback.")

PAGE_OPTIONS = ["Ask", "Settings & Eval", "About the App"]
EVAL_AUTO_REFRESH_SECONDS = 20


def _init_state() -> None:
    st.session_state.setdefault("selected_page", "Ask")
    st.session_state.setdefault("collection_name", HYBRID_BACKEND_NAME)
    st.session_state.setdefault("messages", [])


@st.cache_resource(show_spinner=False)
def _agent(collection_name: str):
    return AutodeskRAGAgent(collection_name=collection_name)


@st.cache_data(ttl=60, show_spinner=False)
def _indexes_ready() -> tuple[bool, bool]:
    return vectorstore_exists(), bm25_index_exists()


def render_ask() -> None:
    st.caption(f"Retrieval backend: {OPTION_3_LABEL}")
    chroma_ready, bm25_ready = _indexes_ready()
    if not chroma_ready or not bm25_ready:
        st.warning("Local Chroma or BM25 indexes are missing. Rebuild with `python scripts/build_retrieval_indexes.py` before expecting grounded answers.")

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input("Ask a question", placeholder="Ask about Autodesk products, subscription options, system requirements, or product comparisons", label_visibility="collapsed")
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

    if st.button("Clear chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()

    if submitted and question.strip():
        question = question.strip()
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Routing, retrieving, checking evidence..."):
                result = _agent(st.session_state.collection_name).answer(question)
            source_mode = "local documents + web search" if result.used_local and result.used_web else "local documents" if result.used_local else "web search" if result.used_web else "error" if result.answer.startswith("Unable") else "no reliable source"
            st.session_state.messages.append({"role": "assistant", "content": result.answer or NO_ANSWER, "sources": result.sources, "source_mode": source_mode, "route_reason": result.route_reason, "web_search_attempted": result.web_search_attempted, "web_query": result.web_query, "web_search_error": result.web_search_error})
        except Exception as exc:
            st.session_state.messages.append({"role": "assistant", "content": f"Unable to answer right now: {exc}", "sources": [], "source_mode": "error"})

    for exchange in reversed(_exchanges(st.session_state.messages)):
        for message in exchange:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    st.caption(f"Answer source: {message.get('source_mode', 'unknown')}")
                    if message.get("route_reason"):
                        st.caption(f"Routing note: {message['route_reason']}")
                    if message.get("web_search_attempted"):
                        st.caption(f"Web search attempted: {message.get('web_query', '')}")
                    if message.get("web_search_error"):
                        st.caption(f"Web search error: {message['web_search_error']}")
                    if message.get("sources"):
                        with st.expander("Sources"):
                            for source in message["sources"]:
                                st.write(source)


def _exchanges(messages: list[dict]) -> list[list[dict]]:
    exchanges: list[list[dict]] = []
    current: list[dict] = []
    for message in messages:
        if message["role"] == "user" and current:
            exchanges.append(current)
            current = []
        current.append(message)
    if current:
        exchanges.append(current)
    return exchanges


def render_settings_eval() -> None:
    settings = get_settings()
    st.subheader("Settings & Eval")
    selected = st.radio("Retrieval configuration", list(COLLECTION_OPTIONS), index=0, horizontal=False)
    st.session_state.collection_name = COLLECTION_OPTIONS[selected]
    st.info("Option 3 uses the existing Autodesk Docling Chroma index for dense semantic retrieval and the persisted local BM25 index for keyword retrieval. Results are fused with Reciprocal Rank Fusion, deduplicated, then expanded with same-document neighbors before the strict adequacy gate.")
    st.caption(f"Context expansion: enabled={settings.context_expansion_enabled}, mode={settings.context_expansion_mode}, neighbor_window=1, max_blocks={settings.context_max_expanded_docs}, max_chars={settings.context_max_chars}.")

    status = _load_json(settings.eval_status_dir / "docling_chroma_bm25_hybrid_status.json")
    results = _load_json(settings.eval_results_dir / "docling_chroma_bm25_hybrid_results.json")
    st.divider()
    st.subheader("Evaluation Metrics")
    st.caption("Saved metrics load from `eval_results/docling_chroma_bm25_hybrid_results.json`. Evaluation runs the fixed 50-question `eval_testset/autodesk_testset.csv` dataset in a background process.")
    if results:
        _render_metrics(results)
        button_label = "Re-run Evaluation"
    else:
        st.warning("No saved evaluation metrics found yet.")
        button_label = "Run Evaluation Metrics"
    if status:
        _render_status(status)
    disabled = bool(status and status.get("status") == "running")
    if st.button(button_label, type="primary", disabled=disabled):
        _start_eval()
        st.success("Evaluation started in the background.")
        st.rerun()


def _render_metrics(results: dict) -> None:
    metrics = results.get("metrics", {})
    st.caption(f"Last run: {results.get('timestamp_utc', 'unknown')} UTC | Questions: {results.get('question_count', 'unknown')}")
    cols = st.columns(5)
    for col, label, key in zip(cols, ["Faithfulness", "Answer Relevance", "Context Precision", "Context Recall"], ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]):
        with col:
            _metric_card(label, metrics.get(key))
    with cols[-1]:
        avg = _num(metrics.get("average_latency"))
        p50 = _num(metrics.get("p50_latency"))
        p99 = _num(metrics.get("p99_latency"))
        st.metric("Latency", "N/A" if avg is None else f"{avg:.1f} | {p50:.1f} | {p99:.1f}", "Avg | P50 | P99")
    if results.get("rows"):
        with st.expander("Evaluation details"):
            st.dataframe(results["rows"], use_container_width=True)


def _metric_card(label: str, value) -> None:
    score = _num(value)
    if score is None:
        st.metric(label, "N/A")
    else:
        band = "Strong" if score >= 0.8 else "Moderate" if score >= 0.6 else "Needs attention"
        st.metric(label, f"{score:.2f}", band)


def _render_status(status: dict) -> None:
    if status.get("status") == "running":
        st.info(f"Evaluation running: {status.get('phase', 'running')}. {status.get('message', '')}")
        total = int(status.get("total") or 50)
        current = int(status.get("current") or 0)
        st.progress(min(max(current / total, 0), 1), text=f"{current} of {total} questions processed")
        components.html(f"<script>setTimeout(() => window.parent.location.reload(), {EVAL_AUTO_REFRESH_SECONDS * 1000});</script>", height=0)
    elif status.get("status") == "complete":
        st.success(f"Last evaluation completed at {status.get('finished_at_utc', 'unknown')} UTC.")
    elif status.get("status") == "error":
        st.error(f"Last evaluation failed: {status.get('error', 'Unknown error')}")


def _start_eval() -> None:
    settings = get_settings()
    settings.eval_status_dir.mkdir(parents=True, exist_ok=True)
    executable = sys.executable
    if sys.platform.startswith("win"):
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        if pythonw.exists():
            executable = str(pythonw)
    kwargs = {"cwd": str(ROOT_DIR), "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.Popen([executable, "-m", "src.evaluation_runner", "--collection-name", HYBRID_BACKEND_NAME], **kwargs)


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _num(value) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def render_about() -> None:
    settings = get_settings()
    st.subheader("About the App")
    st.write("This portfolio app answers Autodesk-related questions with an agentic RAG workflow: a lightweight router classifies the question, local hybrid retrieval searches Chroma and BM25, deterministic neighbor expansion reduces chunk-boundary misses, and a strict adequacy gate refuses unsupported answers.")
    st.write("Local retrieval stays primary. Web search is used when the router sees current/latest/recent signals or when local evidence is weak. The final answer model receives only supplied excerpts, web snippets, and runtime date context.")
    st.table([
        {"Layer": "Retrieval", "Implementation": "Chroma dense semantic search + BM25 keyword search + Reciprocal Rank Fusion"},
        {"Layer": "Context", "Implementation": "Previous/current/next same-document chunk expansion with budget limits"},
        {"Layer": "Generation", "Implementation": f"OpenAI `{settings.openai_model}` with strict source-grounded prompt"},
        {"Layer": "Evaluation", "Implementation": f"50-question golden dataset with five-point LLM judge scoring using `{settings.eval_judge_model}`"},
    ])
    st.info("Portfolio demonstration only. Verify product, pricing, system requirement, and release information directly with Autodesk.")


_init_state()
page = st.radio("Navigation", PAGE_OPTIONS, horizontal=True, label_visibility="collapsed", key="selected_page")
if page == "Ask":
    render_ask()
elif page == "Settings & Eval":
    render_settings_eval()
else:
    render_about()
