"""Streamlit frontend for the Autodesk Agentic RAG app."""

from __future__ import annotations

import csv
import html
import json
import math
import os
import subprocess
import sys
import textwrap
import time
import uuid
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
from src.config import (
    AUTODESK_WEB_MODE,
    COLLECTION_OPTIONS,
    HYBRID_BACKEND_NAME,
    LOCAL_ONLY_MODE,
    OPEN_WEB_MODE,
    OPTION_1_LABEL,
    SEARCH_MODE_OPTIONS,
    get_settings,
)
from src.monitoring import (
    clear_monitoring_logs,
    fetch_recent_interactions,
    log_rag_interaction,
    monitoring_admin_password,
    supabase_monitoring_enabled,
)
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
    :root {
        --autodesk-near-black: rgb(51, 51, 51);
        --autodesk-near-black-hover: rgb(31, 31, 31);
        --autodesk-gold: rgb(212, 175, 55);
        --autodesk-gold-dark: rgb(138, 109, 0);
    }

    div[data-testid="stForm"] {
        border: 1px solid rgba(17, 24, 39, .12);
        border-radius: .5rem;
        padding: 1rem;
        background: #fff;
    }

    div[data-testid="stTextInput"] input {
        font-size: 1.02rem;
        min-height: 3.1rem;
    }

    div[data-testid="stTextInput"] input:focus,
    textarea:focus {
        border-color: var(--autodesk-near-black) !important;
        box-shadow: 0 0 0 1px var(--autodesk-near-black) !important;
    }

    a,
    a:visited {
        color: var(--autodesk-near-black);
    }

    a:hover {
        color: var(--autodesk-near-black-hover);
    }

    div[data-testid="stBaseButton-primary"] button,
    button[data-testid="stBaseButton-primary"],
    button[kind="primary"],
    div[data-testid="stFormSubmitButton"] button {
        background-color: var(--autodesk-near-black) !important;
        border-color: var(--autodesk-near-black) !important;
        color: #fff !important;
    }

    div[data-testid="stBaseButton-primary"] button:hover,
    button[data-testid="stBaseButton-primary"]:hover,
    button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: var(--autodesk-near-black-hover) !important;
        border-color: var(--autodesk-near-black-hover) !important;
        color: #fff !important;
    }

    div[data-testid="stBaseButton-primary"] button:focus,
    button[data-testid="stBaseButton-primary"]:focus,
    button[kind="primary"]:focus,
    div[data-testid="stFormSubmitButton"] button:focus {
        box-shadow: 0 0 0 .2rem rgba(51, 51, 51, .18) !important;
    }

    div[data-testid="stChatMessageAvatarUser"],
    div[data-testid="stChatMessageAvatarAssistant"] {
        background-color: var(--autodesk-near-black) !important;
    }

    input[type="radio"] {
        accent-color: var(--autodesk-gold);
    }

    div[role="radiogroup"] label {
        color: #111827 !important;
    }

    div[role="radiogroup"] label [aria-checked="true"],
    div[role="radiogroup"] label:has(input[type="radio"]:checked) > div:first-child {
        border-color: var(--autodesk-gold) !important;
        background-color: var(--autodesk-gold) !important;
    }

    div[role="radiogroup"] label [aria-checked="true"] svg,
    div[role="radiogroup"] label:has(input[type="radio"]:checked) svg {
        fill: var(--autodesk-gold) !important;
        color: var(--autodesk-gold) !important;
    }

    div[role="radiogroup"] label:hover [aria-checked="false"],
    div[role="radiogroup"] label:hover > div:first-child {
        border-color: var(--autodesk-gold-dark) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Autodesk Agentic RAG")
st.caption("Hybrid BM25 keyword plus Chroma vector retrieval over an Autodesk corpus, with selectable web-search policy.")

PAGE_OPTIONS = ["Ask", "Settings & Eval", "Monitoring", "About the App"]
EVAL_AUTO_REFRESH_SECONDS = 20
NO_ANSWER_TEXT = "I could not find a reliable answer in the available documents or web sources."


def get_query_param(name: str, default: str) -> str:
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value)


def _init_state() -> None:
    query_page = get_query_param("page", "Ask")
    if query_page not in PAGE_OPTIONS:
        query_page = "Ask"
    st.session_state.setdefault("selected_page", query_page)

    query_mode = get_query_param("mode", OPTION_1_LABEL)
    if query_mode not in SEARCH_MODE_OPTIONS:
        query_mode = OPTION_1_LABEL
    st.session_state.setdefault("search_mode_label", query_mode)
    st.session_state.setdefault("search_mode", SEARCH_MODE_OPTIONS[query_mode])
    st.session_state.setdefault("collection_name", COLLECTION_OPTIONS[query_mode])
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    _normalize_search_mode_state()


def sync_query_state(page: str | None = None, mode_label: str | None = None) -> None:
    if page and st.query_params.get("page") != page:
        st.query_params["page"] = page
    if mode_label and st.query_params.get("mode") != mode_label:
        st.query_params["mode"] = mode_label


def _mode_label_for_search_mode(search_mode: str | None) -> str:
    for label, mode in SEARCH_MODE_OPTIONS.items():
        if mode == search_mode:
            return label
    return OPTION_1_LABEL


def _retrieval_backend_label(search_mode: str | None) -> str:
    if search_mode == AUTODESK_WEB_MODE:
        return "docling_chroma_bm25_hybrid_autodesk_web"
    if search_mode == OPEN_WEB_MODE:
        return "docling_chroma_bm25_hybrid_open_web"
    return "docling_chroma_bm25_hybrid_local_only"


def _normalize_search_mode_state() -> str:
    search_mode = st.session_state.get("search_mode")
    mode_label = st.session_state.get("search_mode_label")
    if search_mode in set(SEARCH_MODE_OPTIONS.values()):
        mode_label = _mode_label_for_search_mode(search_mode)
    elif mode_label in SEARCH_MODE_OPTIONS:
        search_mode = SEARCH_MODE_OPTIONS[mode_label]
    else:
        mode_label = OPTION_1_LABEL
        search_mode = SEARCH_MODE_OPTIONS[mode_label]

    st.session_state.search_mode_label = mode_label
    st.session_state.search_mode = search_mode
    st.session_state.collection_name = COLLECTION_OPTIONS[mode_label]
    return mode_label


def on_page_change() -> None:
    selected_page = st.session_state.get("selected_page", "Ask")
    if selected_page not in PAGE_OPTIONS:
        selected_page = "Ask"
    sync_query_state(page=selected_page, mode_label=_normalize_search_mode_state())


def on_search_mode_change() -> None:
    selected_label = st.session_state.get("search_mode_label", OPTION_1_LABEL)
    if selected_label not in SEARCH_MODE_OPTIONS:
        selected_label = OPTION_1_LABEL
    st.session_state.search_mode_label = selected_label
    st.session_state.search_mode = SEARCH_MODE_OPTIONS[selected_label]
    st.session_state.collection_name = COLLECTION_OPTIONS[selected_label]
    sync_query_state(mode_label=selected_label)


@st.cache_resource(show_spinner=False)
def _agent(collection_name: str, search_mode: str):
    return AutodeskRAGAgent(collection_name=collection_name, search_mode=search_mode)


@st.cache_data(ttl=60, show_spinner=False)
def _indexes_ready() -> tuple[bool, bool]:
    return vectorstore_exists(), bm25_index_exists()


def render_ask() -> None:
    mode_label = _normalize_search_mode_state()
    sync_query_state(page="Ask", mode_label=mode_label)
    st.caption(f"Search mode: {mode_label}")
    st.caption("Local backend: Docling + Chroma + BM25 Hybrid Search")
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
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Routing, retrieving, checking evidence..."):
                result = _agent(st.session_state.collection_name, st.session_state.search_mode).answer(question)
            latency_total_sec = time.perf_counter() - started
            source_mode = _source_mode_label(result.used_local, result.used_web, st.session_state.search_mode)
            answered_mode_label = _mode_label_for_search_mode(st.session_state.search_mode)
            log_rag_interaction(
                _rag_monitoring_event(
                    question=question,
                    request_id=request_id,
                    final_response=result.answer or NO_ANSWER,
                    success=True,
                    latency_total_sec=latency_total_sec,
                    result=result,
                    source_mode=source_mode,
                    selected_mode_label=answered_mode_label,
                )
            )
            st.session_state.messages.append({"role": "assistant", "content": result.answer or NO_ANSWER, "sources": result.sources, "source_mode": source_mode, "route_reason": result.route_reason, "web_search_attempted": result.web_search_attempted, "web_query": result.web_query, "web_search_error": result.web_search_error, "search_mode_label": answered_mode_label})
        except Exception as exc:
            latency_total_sec = time.perf_counter() - started
            log_rag_interaction(
                _rag_monitoring_event(
                    question=question,
                    request_id=request_id,
                    final_response=None,
                    success=False,
                    latency_total_sec=latency_total_sec,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            )
            st.session_state.messages.append({"role": "assistant", "content": f"Unable to answer right now: {exc}", "sources": [], "source_mode": "error"})

    for exchange in reversed(_exchanges(st.session_state.messages)):
        for message in exchange:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    st.caption(f"Answer source: {message.get('source_mode', 'unknown')}")
                    if message.get("search_mode_label"):
                        st.caption(f"Search mode: {message['search_mode_label']}")
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


def _source_mode_label(used_local: bool, used_web: bool, search_mode: str) -> str:
    if used_local and used_web and search_mode == AUTODESK_WEB_MODE:
        return "local documents + autodesk.com web search"
    if used_local and used_web and search_mode == OPEN_WEB_MODE:
        return "local documents + open web search"
    if used_local:
        return "local documents"
    if used_web and search_mode == AUTODESK_WEB_MODE:
        return "autodesk.com web search"
    if used_web and search_mode == OPEN_WEB_MODE:
        return "open web search"
    return "no reliable source"


def _rag_monitoring_event(
    *,
    question: str,
    request_id: str,
    final_response: str | None,
    success: bool,
    latency_total_sec: float,
    result=None,
    source_mode: str | None = None,
    selected_mode_label: str | None = None,
    error: str | None = None,
    error_type: str | None = None,
) -> dict:
    settings = get_settings()
    sources = list(getattr(result, "sources", []) or []) if result is not None else []
    contexts = list(getattr(result, "contexts", []) or []) if result is not None else []
    no_answer = bool(final_response == NO_ANSWER_TEXT)
    web_error = str(getattr(result, "web_search_error", "") or "") if result is not None else ""
    web_search_attempted = bool(getattr(result, "web_search_attempted", False)) if result is not None else False
    return {
        "app_name": "autodesk-rag",
        "app_environment": os.getenv("APP_ENVIRONMENT", "streamlit"),
        "session_id": st.session_state.get("session_id"),
        "request_id": request_id,
        "question": question,
        "final_response": final_response,
        "retrieval_backend": _retrieval_backend_label(st.session_state.get("search_mode")),
        "router_decision": getattr(result, "route_reason", None) if result is not None else None,
        "used_local": bool(getattr(result, "used_local", False)) if result is not None else False,
        "used_web": bool(getattr(result, "used_web", False)) if result is not None else False,
        "web_fallback_reason": web_error or None,
        "adequacy_answerable": None if result is None else not no_answer,
        "no_answer": no_answer,
        "source_count": len(sources),
        "retrieved_chunk_count": len(contexts) if contexts else None,
        "expanded_context_chars": sum(len(context or "") for context in contexts) if contexts else None,
        "sources": sources,
        "latency_total_sec": latency_total_sec,
        "latency_router_sec": getattr(result, "latency_router_sec", None) if result is not None else None,
        "latency_retrieval_sec": getattr(result, "latency_retrieval_sec", None) if result is not None else None,
        "latency_expansion_sec": getattr(result, "latency_expansion_sec", None) if result is not None else None,
        "latency_adequacy_sec": getattr(result, "latency_adequacy_sec", None) if result is not None else None,
        "latency_web_sec": getattr(result, "latency_web_sec", None) if result is not None else None,
        "latency_generation_sec": getattr(result, "latency_generation_sec", None) if result is not None else None,
        "model_name": settings.openai_model,
        "embedding_model_name": settings.openai_embedding_model,
        "prompt_tokens": getattr(result, "prompt_tokens", None) if result is not None else None,
        "completion_tokens": getattr(result, "completion_tokens", None) if result is not None else None,
        "total_tokens": getattr(result, "total_tokens", None) if result is not None else None,
        "estimated_cost_usd": getattr(result, "estimated_cost_usd", None) if result is not None else None,
        "success": success,
        "error": error,
        "error_type": error_type,
        "extra": {
            "langsmith_project": settings.langsmith_project or os.getenv("LANGSMITH_PROJECT"),
            "selected_ui_option": selected_mode_label or _mode_label_for_search_mode(st.session_state.get("search_mode")),
            "search_mode": st.session_state.get("search_mode"),
            "local_retrieval_backend": st.session_state.get("collection_name", HYBRID_BACKEND_NAME),
            "answer_source_type": source_mode,
            "web_search_attempted": web_search_attempted,
            "web_query": getattr(result, "web_query", None) if result is not None else None,
        },
    }


def render_settings_eval() -> None:
    settings = get_settings()
    st.subheader("Settings & Eval")
    labels = list(SEARCH_MODE_OPTIONS)
    current_label = _normalize_search_mode_state()
    if st.session_state.get("search_mode_label") != current_label:
        st.session_state.search_mode_label = current_label
    st.radio(
        "Retrieval configuration",
        labels,
        index=labels.index(current_label) if current_label in labels else 0,
        horizontal=False,
        key="search_mode_label",
        on_change=on_search_mode_change,
    )
    selected = st.session_state.get("search_mode_label", OPTION_1_LABEL)
    st.session_state.search_mode = SEARCH_MODE_OPTIONS[selected]
    st.session_state.collection_name = COLLECTION_OPTIONS[selected]
    sync_query_state(page="Settings & Eval", mode_label=selected)
    st.info(_mode_explanation(st.session_state.search_mode))
    st.caption("All three options use the same local Docling + Chroma + BM25 Hybrid Search backend. The radio button controls whether web evidence is disabled, scoped to autodesk.com, or open web.")
    st.caption(f"Context expansion: enabled={settings.context_expansion_enabled}, mode={settings.context_expansion_mode}, neighbor_window=1, max_blocks={settings.context_max_expanded_docs}, max_chars={settings.context_max_chars}.")
    st.caption(
        f"Cross-encoder reranker: enabled={settings.reranker_enabled}, "
        f"model={settings.reranker_model}, top_n={settings.reranker_top_n}."
    )

    status = _load_json(settings.eval_status_dir / _eval_status_filename(st.session_state.search_mode))
    results = _load_json(settings.eval_results_dir / _eval_results_filename(st.session_state.search_mode))
    st.divider()
    st.subheader("Evaluation Metrics")
    st.caption(
        f"Saved metrics load from `eval_results/{_eval_results_filename(st.session_state.search_mode)}`. "
        "Evaluation runs the fixed 50-question `eval_testset/autodesk_testset.csv` dataset in a background process."
    )
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
        _start_eval(st.session_state.search_mode)
        st.success("Evaluation started in the background.")
        st.rerun()


def _mode_explanation(search_mode: str) -> str:
    if search_mode == LOCAL_ONLY_MODE:
        return (
            "Option 1: Local Document Search uses only indexed local Autodesk corpus documents. "
            "The router does not evaluate whether web search is suitable, and web search is disabled."
        )
    if search_mode == AUTODESK_WEB_MODE:
        return (
            "Option 2: Local Document Search + Autodesk.com uses local documents first and always incorporates "
            "SerpAPI Google results restricted to autodesk.com pages."
        )
    return (
        "Option 3: Local Document Search + Open Web Search uses local documents first and always incorporates broader "
        "web search. Open web search is capped at three results to keep latency and noise lower."
    )


def _eval_results_filename(search_mode: str) -> str:
    if search_mode == AUTODESK_WEB_MODE:
        return "docling_chroma_bm25_hybrid_autodesk_web_results.json"
    if search_mode == OPEN_WEB_MODE:
        return "docling_chroma_bm25_hybrid_open_web_results.json"
    return "docling_chroma_bm25_hybrid_results.json"


def _eval_status_filename(search_mode: str) -> str:
    if search_mode == AUTODESK_WEB_MODE:
        return "docling_chroma_bm25_hybrid_autodesk_web_status.json"
    if search_mode == OPEN_WEB_MODE:
        return "docling_chroma_bm25_hybrid_open_web_status.json"
    return "docling_chroma_bm25_hybrid_status.json"


def _render_metrics(results: dict) -> None:
    metrics = results.get("metrics", {})
    _metric_styles()
    st.caption(f"Last run: {results.get('timestamp_utc', 'unknown')} UTC | Questions: {results.get('question_count', 'unknown')}")
    if results.get("dataset_name"):
        st.caption(f"LangSmith dataset: {results['dataset_name']}")
    if results.get("experiment_url"):
        st.markdown(f"[Open LangSmith experiment]({results['experiment_url']})")
    cols = st.columns(5)
    for col, label, key in zip(cols, ["Faithfulness", "Answer Relevance", "Context Precision", "Context Recall"], ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]):
        with col:
            _metric_card(label, metrics.get(key))
    with cols[-1]:
        avg = _num(metrics.get("average_latency"))
        p50 = _num(metrics.get("p50_latency"))
        p99 = _num(metrics.get("p99_latency"))
        _latency_card(avg, p50, p99)
    if results.get("rows"):
        with st.expander("Evaluation details"):
            _render_evaluation_details(results["rows"])


def _render_evaluation_details(rows: list[dict]) -> None:
    ordered_rows = _ordered_eval_rows(rows)
    st.caption("Rows are sorted to match `eval_testset/autodesk_testset.csv`; the first six rows are the required reviewer questions.")
    st.markdown(_evaluation_details_table(ordered_rows), unsafe_allow_html=True)


def _ordered_eval_rows(rows: list[dict]) -> list[dict]:
    order = _testset_question_order()
    return sorted(rows, key=lambda row: order.get(str(row.get("inputs.question") or ""), 10_000))


@st.cache_data(show_spinner=False)
def _testset_question_order() -> dict[str, int]:
    path = get_settings().eval_testset_path
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            return {str(row.get("question") or "").strip(): index for index, row in enumerate(reader, start=1)}
    except Exception:
        return {}


def _evaluation_details_table(rows: list[dict]) -> str:
    table_rows = []
    order = _testset_question_order()
    for fallback_index, row in enumerate(rows, start=1):
        question = str(row.get("inputs.question") or "")
        row_number = order.get(question, fallback_index)
        answer = str(row.get("outputs.answer") or "")
        reference = str(row.get("reference.answer") or "")
        faithfulness = _format_score(row.get("feedback.faithfulness"))
        relevance = _format_score(row.get("feedback.answer_relevancy"))
        precision = _format_score(row.get("feedback.context_precision"))
        recall = _format_score(row.get("feedback.context_recall"))
        latency = _format_seconds(row.get("outputs.execution_time") or row.get("execution_time"))
        source_mode = _format_source_mode(row)
        error = str(row.get("outputs.error") or row.get("error") or "")
        if error:
            error_block = f"<div class=\"eval-error\">{html.escape(error)}</div>"
        else:
            error_block = ""
        table_rows.append(
            "<tr>"
            f"<td class=\"eval-num\">{row_number}</td>"
            f"<td class=\"eval-question\">{html.escape(question)}</td>"
            f"<td class=\"eval-answer\">{html.escape(answer)}</td>"
            f"<td class=\"eval-reference\">{html.escape(reference)}</td>"
            f"<td class=\"eval-scores\">"
            f"<div>Faithfulness: {faithfulness}</div>"
            f"<div>Relevance: {relevance}</div>"
            f"<div>Precision: {precision}</div>"
            f"<div>Recall: {recall}</div>"
            f"<div>Latency: {latency}</div>"
            f"<div>Evidence: {source_mode}</div>"
            f"{error_block}"
            "</td>"
            "</tr>"
        )
    prefix = textwrap.dedent(
        """
        <style>
        .eval-review-table {
            border-collapse: collapse;
            width: 100%;
            table-layout: fixed;
            font-family: "Source Sans Pro", Arial, sans-serif;
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .eval-review-table * {
            font-family: inherit !important;
            font-size: inherit !important;
            line-height: inherit !important;
        }
        .eval-review-table th {
            background: #f3f4f6;
            border: 1px solid #d1d5db;
            color: #111827;
            font-weight: 700;
            padding: 0.7rem;
            text-align: left;
            vertical-align: top;
        }
        .eval-review-table td {
            border: 1px solid #d1d5db;
            color: #111827;
            padding: 0.85rem;
            vertical-align: top;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            word-break: normal;
        }
        .eval-num {
            width: 3.5rem;
            text-align: center;
            font-weight: 700;
        }
        .eval-question {
            width: 18%;
            font-weight: 650;
        }
        .eval-answer {
            width: 34%;
        }
        .eval-reference {
            width: 27%;
            color: #374151;
        }
        .eval-scores {
            width: 16%;
        }
        .eval-error {
            color: #991b1b;
            font-weight: 650;
            margin-top: 0.5rem;
        }
        </style>
        <table class="eval-review-table">
            <thead>
                <tr>
                    <th class="eval-num">#</th>
                    <th class="eval-question">Question</th>
                    <th class="eval-answer">App Answer</th>
                    <th class="eval-reference">Reference Answer</th>
                    <th class="eval-scores">Scores</th>
                </tr>
            </thead>
            <tbody>
        """
    ).strip()
    suffix = textwrap.dedent(
        """
            </tbody>
        </table>
        """
    ).strip()
    return f"{prefix}\n{'\n'.join(table_rows)}\n{suffix}"


def _format_score(value) -> str:
    score = _num(value)
    return "N/A" if score is None else f"{score:.2f}"


def _format_seconds(value) -> str:
    seconds = _num(value)
    return "N/A" if seconds is None else f"{seconds:.1f}s"


def _format_source_mode(row: dict) -> str:
    used_local = bool(row.get("outputs.used_local"))
    used_web = bool(row.get("outputs.used_web"))
    if used_local and used_web:
        return "local + web"
    if used_local:
        return "local"
    if used_web:
        return "web"
    return "no reliable source"


def _metric_styles() -> None:
    st.markdown(
        """
        <style>
        .rag-metric-label {
            color: #111827;
            font-size: 0.92rem;
            line-height: 1.25;
            margin-bottom: 0.35rem;
        }
        .rag-metric-value {
            color: #111827;
            font-size: 2.15rem;
            line-height: 1.05;
            margin-bottom: 0.35rem;
        }
        .rag-metric-status {
            font-size: 0.95rem;
            font-weight: 500;
            line-height: 1.25;
        }
        .rag-metric-good {
            color: #15803d;
        }
        .rag-metric-moderate {
            color: #8a6d00;
        }
        .rag-metric-low {
            color: #991b1b;
        }
        .rag-metric-muted {
            color: #6b7280;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value) -> None:
    score = _num(value)
    if score is None:
        _custom_metric(label, "N/A", "No score", "rag-metric-muted", "-")
        return
    band, css_class, icon = _quality_band(score)
    _custom_metric(label, f"{score:.2f}", band, css_class, icon)


def _latency_card(avg: float | None, p50: float | None, p99: float | None) -> None:
    if avg is None:
        _custom_metric("Latency", "N/A", "No score", "rag-metric-muted", "-")
        return
    band, css_class, icon = _latency_band(avg)
    p50_text = "N/A" if p50 is None else f"{p50:.1f}"
    p99_text = "N/A" if p99 is None else f"{p99:.1f}"
    _custom_metric("Latency", f"{avg:.1f} | {p50_text} | {p99_text}", f"{band} Avg | P50 | P99", css_class, icon)


def _custom_metric(label: str, value: str, status: str, css_class: str, icon: str) -> None:
    st.markdown(
        f"""
        <div class="rag-metric">
            <div class="rag-metric-label">{label}</div>
            <div class="rag-metric-value">{value}</div>
            <div class="rag-metric-status {css_class}">{icon} {status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _quality_band(score: float) -> tuple[str, str, str]:
    if score >= 0.8:
        return "Strong", "rag-metric-good", "✓"
    if score >= 0.6:
        return "Moderate", "rag-metric-moderate", "●"
    return "Needs attention", "rag-metric-low", "!"


def _latency_band(avg_seconds: float) -> tuple[str, str, str]:
    if avg_seconds <= 10:
        return "Strong", "rag-metric-good", "✓"
    if avg_seconds <= 25:
        return "Moderate", "rag-metric-moderate", "●"
    return "Needs attention", "rag-metric-low", "!"


def _render_status(status: dict) -> None:
    if status.get("status") == "running":
        phase = str(status.get("phase") or "running").replace("_", " ").title()
        message = status.get("message", "")
        st.info(f"Evaluation running: {phase}. {message}")
        total = int(status.get("total") or 50)
        current = int(status.get("current") or 0)
        st.progress(min(max(current / total, 0), 1), text=f"{current} of {total} questions processed")
        if status.get("question"):
            st.caption(f"Current question: {status['question']}")
        if status.get("started_at_utc"):
            st.caption(f"Started: {status['started_at_utc']} UTC | Elapsed: {_elapsed_since(status['started_at_utc'])}")
        if status.get("updated_at_utc"):
            st.caption(f"Last status update: {status['updated_at_utc']} UTC")
        if status.get("execution_time") is not None:
            st.caption(f"Last question latency: {float(status['execution_time']):.1f} seconds")
        st.caption(
            f"This dashboard refreshes every {EVAL_AUTO_REFRESH_SECONDS} seconds while evaluation is running. "
            "Use Refresh now if your browser pauses background timers."
        )
        if st.button("Refresh now", type="secondary"):
            st.rerun()
        components.html(f"<script>setTimeout(() => window.parent.location.reload(), {EVAL_AUTO_REFRESH_SECONDS * 1000});</script>", height=0)
    elif status.get("status") == "complete":
        st.success(f"Last evaluation completed at {status.get('finished_at_utc', 'unknown')} UTC.")
        if status.get("experiment_url"):
            st.markdown(f"[Open LangSmith experiment]({status['experiment_url']})")
    elif status.get("status") == "error":
        st.error(f"Last evaluation failed: {status.get('error', 'Unknown error')}")


def _start_eval(search_mode: str) -> None:
    settings = get_settings()
    settings.eval_status_dir.mkdir(parents=True, exist_ok=True)
    started = _now_utc()
    _write_eval_status(
        settings.eval_status_dir / _eval_status_filename(search_mode),
        {
            "status": "running",
            "phase": "launching",
            "message": "Launching background LangSmith evaluator.",
            "current": 0,
            "total": 50,
            "started_at_utc": started,
            "updated_at_utc": started,
        },
    )
    executable = sys.executable
    if sys.platform.startswith("win"):
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        if pythonw.exists():
            executable = str(pythonw)
    kwargs = {"cwd": str(ROOT_DIR), "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.Popen(
        [
            executable,
            "-m",
            "src.evaluation_runner",
            "--collection-name",
            HYBRID_BACKEND_NAME,
            "--search-mode",
            search_mode,
        ],
        **kwargs,
    )


def _write_eval_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _elapsed_since(timestamp: str) -> str:
    try:
        started = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        elapsed_seconds = max(int((datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds()), 0)
    except Exception:
        return "unknown"
    minutes, seconds = divmod(elapsed_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _num(value) -> float | None:
    try:
        number = float(value)
        return None if math.isnan(number) else number
    except Exception:
        return None


def render_monitoring() -> None:
    st.subheader("Monitoring")
    st.write("Runtime monitoring summarizes recent RAG interactions logged to Supabase, including retrieval choices, source usage, latency, no-answer outcomes, and errors.")
    st.warning(
        "This demo logs submitted questions, generated responses, retrieval metadata, latency, and errors for monitoring and debugging. "
        "Do not enter sensitive, private, or confidential information."
    )

    enabled = supabase_monitoring_enabled()
    st.caption(f"Supabase monitoring configured: {'Yes' if enabled else 'No'}")
    if not enabled:
        st.info("Monitoring is not configured for this environment. The Ask tab still works normally; add SUPABASE_URL and SUPABASE_KEY in Streamlit secrets or environment variables to enable logging.")
        return

    rows = fetch_recent_interactions(limit=1000)
    if not rows:
        st.info("No monitoring rows found yet. Ask a question to populate this dashboard.")
        _render_admin_controls()
        return

    import pandas as pd

    df = pd.DataFrame(rows)
    if "created_at" in df:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    for column in ("latency_total_sec", "source_count", "total_tokens", "estimated_cost_usd"):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("success", "used_web", "no_answer", "adequacy_answerable"):
        if column in df:
            df[column] = df[column].astype("boolean")
    filtered = _filter_monitoring_frame(df)
    if filtered.empty:
        st.info("No rows match the selected filters.")
        _render_admin_controls()
        return

    _render_monitoring_metrics(filtered)
    st.divider()
    _render_monitoring_charts(filtered)
    st.divider()
    _render_monitoring_tables(filtered)
    _render_monitoring_latency_diagnostics(filtered)
    _render_admin_controls()


def _filter_monitoring_frame(df):
    import pandas as pd

    with st.expander("Filters", expanded=True):
        cols = st.columns(5)
        filtered = df.copy()
        if "created_at" in filtered and filtered["created_at"].notna().any():
            min_date = filtered["created_at"].min().date()
            max_date = filtered["created_at"].max().date()
            date_range = cols[0].date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                filtered = filtered[
                    (filtered["created_at"].dt.date >= start_date)
                    & (filtered["created_at"].dt.date <= end_date)
                ]

        backends = sorted(str(value) for value in filtered.get("retrieval_backend", pd.Series(dtype=str)).dropna().unique())
        backend_choice = cols[1].selectbox("Backend", ["All", *backends])
        if backend_choice != "All":
            filtered = filtered[filtered["retrieval_backend"].astype(str) == backend_choice]

        success_choice = cols[2].selectbox("Outcome", ["All", "Success", "Error"])
        if "success" in filtered:
            if success_choice == "Success":
                filtered = filtered[filtered["success"].fillna(True) == True]
            elif success_choice == "Error":
                filtered = filtered[filtered["success"].fillna(True) == False]

        web_choice = cols[3].selectbox("Used web", ["All", "Yes", "No"])
        if "used_web" in filtered:
            if web_choice == "Yes":
                filtered = filtered[filtered["used_web"].fillna(False) == True]
            elif web_choice == "No":
                filtered = filtered[filtered["used_web"].fillna(False) == False]

        no_answer_choice = cols[4].selectbox("No-answer", ["All", "Yes", "No"])
        if "no_answer" in filtered:
            if no_answer_choice == "Yes":
                filtered = filtered[filtered["no_answer"].fillna(False) == True]
            elif no_answer_choice == "No":
                filtered = filtered[filtered["no_answer"].fillna(False) == False]

    return filtered


def _render_monitoring_metrics(df) -> None:
    latency = df["latency_total_sec"] if "latency_total_sec" in df else None
    token_data = df["total_tokens"] if "total_tokens" in df else None
    cost_data = df["estimated_cost_usd"] if "estimated_cost_usd" in df else None
    no_answer = df["no_answer"].fillna(False) if "no_answer" in df else None
    used_web = df["used_web"].fillna(False) if "used_web" in df else None
    success = df["success"].fillna(True) if "success" in df else None
    adequacy = df["adequacy_answerable"].dropna() if "adequacy_answerable" in df else None
    source_count = df["source_count"] if "source_count" in df else None
    cols = st.columns(5)
    cols[0].metric("Total logged questions", f"{len(df):,}")
    cols[1].metric("Average latency", _format_metric_seconds(latency.mean() if latency is not None else None))
    cols[2].metric("P50 latency", _format_metric_seconds(latency.quantile(0.50) if latency is not None else None))
    cols[3].metric("P99 latency", _format_metric_seconds(latency.quantile(0.99) if latency is not None else None))
    cols[4].metric("No-answer rate", _format_percent(no_answer.mean() if no_answer is not None else None))

    cols = st.columns(5)
    cols[0].metric("Web fallback rate", _format_percent(used_web.mean() if used_web is not None else None))
    cols[1].metric("Adequacy pass rate", _format_percent(adequacy.mean() if adequacy is not None and not adequacy.empty else None))
    cols[2].metric("Error rate", _format_percent((success == False).mean() if success is not None else None))
    cols[3].metric("Average source count", _format_metric_number(source_count.mean() if source_count is not None else None))
    if token_data is not None and token_data.notna().any():
        cols[4].metric("Average tokens", _format_metric_number(token_data.mean(), decimals=0))
    elif cost_data is not None and cost_data.notna().any():
        cols[4].metric("Total estimated cost", f"${cost_data.sum():.4f}")
    else:
        cols[4].metric("Average tokens", "N/A")

    if cost_data is not None and cost_data.notna().any() and token_data is not None and token_data.notna().any():
        st.metric("Total estimated cost", f"${cost_data.sum():.4f}")


def _render_monitoring_charts(df) -> None:
    import pandas as pd

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.caption("Backend usage")
        backend_series = df["retrieval_backend"] if "retrieval_backend" in df else pd.Series(dtype=str)
        _render_donut_chart(
            backend_series.fillna("unknown").value_counts(),
            "Backend",
            "Requests",
            label_map={
                "docling_chroma_bm25_hybrid_local_only": "Local only",
                "docling_chroma_bm25_hybrid_autodesk_web": "Local + Autodesk.com",
                "docling_chroma_bm25_hybrid_open_web": "Local + open web",
                "docling_chroma_bm25_hybrid": "Local only",
            },
            color_map={
                "Local only": "#ef4444",
                "Local + Autodesk.com": "#8b5cf6",
                "Local + open web": "#fbbf24",
            },
        )
    with chart_cols[1]:
        st.caption("Used web vs not used web")
        web_series = df["used_web"] if "used_web" in df else pd.Series(dtype=bool)
        _render_donut_chart(
            web_series.fillna(False).map({True: "Used web", False: "No web"}).value_counts(),
            "Web usage",
            "Requests",
        )


def _render_donut_chart(
    counts,
    label_name: str,
    count_name: str,
    label_map: dict[str, str] | None = None,
    color_map: dict[str, str] | None = None,
) -> None:
    import altair as alt
    import pandas as pd

    data = counts.reset_index()
    data.columns = ["Raw label", count_name]
    data[label_name] = data["Raw label"].map(label_map or {}).fillna(data["Raw label"].astype(str))
    total = float(data[count_name].sum() or 1)
    data["Percent"] = data[count_name] / total
    chart = (
        alt.Chart(data)
        .mark_arc(innerRadius=55, outerRadius=105)
        .encode(
            theta=alt.Theta(f"{count_name}:Q"),
            color=_donut_color_encoding(label_name, color_map),
            tooltip=[
                alt.Tooltip(f"{label_name}:N", title=label_name),
                alt.Tooltip(f"{count_name}:Q", title=count_name),
                alt.Tooltip("Percent:Q", title="Percent", format=".1%"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)
    legend_table = data[[label_name, count_name, "Percent"]].copy()
    legend_table["Percent"] = legend_table["Percent"].map(lambda value: f"{value:.1%}")
    st.dataframe(legend_table, use_container_width=True, hide_index=True, height=150)


def _donut_color_encoding(label_name: str, color_map: dict[str, str] | None):
    import altair as alt

    if not color_map:
        return alt.Color(f"{label_name}:N", legend=alt.Legend(title=None))
    return alt.Color(
        f"{label_name}:N",
        scale=alt.Scale(domain=list(color_map), range=list(color_map.values())),
        legend=alt.Legend(title=None),
    )


def _render_monitoring_tables(df) -> None:
    table_cols = st.columns([1, 2])
    with table_cols[0]:
        st.subheader("Top Source Documents")
        top_sources = _top_source_documents_table(df)
        if top_sources.empty:
            st.info("No source metadata available yet.")
        else:
            st.dataframe(top_sources, use_container_width=True, hide_index=True, height=360)
    with table_cols[1]:
        _render_recent_interactions(df, height=360)


def _top_source_documents_table(df):
    import pandas as pd

    source_counts: dict[str, int] = {}
    for names in df.get("top_source_names", []):
        if isinstance(names, list):
            for name in names:
                source_counts[str(name)] = source_counts.get(str(name), 0) + 1
    if not source_counts:
        return pd.DataFrame(columns=["Rank", "Document Title", "Hits"])
    top_sources = (
        pd.Series(source_counts)
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top_sources.columns = ["Document Title", "Hits"]
    top_sources.insert(0, "Rank", range(1, len(top_sources) + 1))
    return top_sources


def _render_monitoring_latency_diagnostics(df) -> None:
    import altair as alt
    import pandas as pd

    latency_columns = [
        ("Router", "latency_router_sec"),
        ("Retrieval", "latency_retrieval_sec"),
        ("Expansion", "latency_expansion_sec"),
        ("Adequacy", "latency_adequacy_sec"),
        ("Web", "latency_web_sec"),
        ("Generation", "latency_generation_sec"),
    ]
    available_latency = {
        label: column
        for label, column in latency_columns
        if column in df and df[column].notna().any()
    }
    if not available_latency and ("latency_total_sec" not in df or not df["latency_total_sec"].notna().any()):
        return

    st.divider()
    st.subheader("Latency Diagnostics")
    st.caption("These charts are more useful than time trends for small datasets because they show where each RAG request spends time.")
    cols = st.columns(2)
    with cols[0]:
        st.caption("Average latency by pipeline stage")
        if available_latency:
            stage_order = [label for label, _ in latency_columns]
            stage_means = pd.DataFrame(
                [
                    {"Stage": label, "Average seconds": df[column].mean()}
                    for label, column in latency_columns
                    if label in available_latency
                ]
            )
            chart = (
                alt.Chart(stage_means)
                .mark_bar()
                .encode(
                    x=alt.X("Stage:N", sort=stage_order, title=None),
                    y=alt.Y("Average seconds:Q", title="Average seconds"),
                    tooltip=["Stage:N", alt.Tooltip("Average seconds:Q", format=".2f")],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Stage-level latency is available only for newer logged interactions.")

    with cols[1]:
        st.caption("Total latency over time")
        if (
            "created_at" in df
            and "latency_total_sec" in df
            and df["created_at"].notna().any()
            and df["latency_total_sec"].notna().any()
        ):
            latency_time = df.dropna(subset=["created_at", "latency_total_sec"]).sort_values("created_at").copy()
            latency_time["Question"] = latency_time["question"].fillna("").map(lambda value: _truncate_text(str(value), 80)) if "question" in latency_time else "Question"
            chart = (
                alt.Chart(latency_time)
                .mark_line(point=True)
                .encode(
                    x=alt.X("created_at:T", title="Time"),
                    y=alt.Y("latency_total_sec:Q", title="Total latency (seconds)"),
                    tooltip=[
                        alt.Tooltip("created_at:T", title="Time"),
                        alt.Tooltip("latency_total_sec:Q", title="Latency", format=".2f"),
                        alt.Tooltip("Question:N", title="Question"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Time-series latency needs created_at and total latency values.")


def _render_recent_interactions(df, height: int | None = None) -> None:
    columns = [
        "created_at",
        "retrieval_backend",
        "router_decision",
        "used_web",
        "adequacy_answerable",
        "no_answer",
        "source_count",
        "latency_total_sec",
        "success",
        "question",
        "final_response",
        "error_type",
    ]
    visible_columns = [column for column in columns if column in df]
    display = df[visible_columns].copy()
    for column in ("question", "final_response"):
        if column in display:
            display[column] = display[column].fillna("").map(lambda value: _truncate_text(str(value), 220))
    st.subheader("Recent Interactions")
    st.markdown(_recent_interactions_table_html(display, max_height=height or 360), unsafe_allow_html=True)


def _recent_interactions_table_html(display, max_height: int = 360) -> str:
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in display.columns)
    rows = []
    for _, row in display.iterrows():
        cells = []
        for column in display.columns:
            value = "" if row[column] is None else str(row[column])
            cells.append(f"<td><div class=\"recent-cell-text\">{html.escape(value)}</div></td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return textwrap.dedent(
        f"""
        <style>
        .recent-table-wrap {{
            max-height: {max_height}px;
            overflow: auto;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
        }}
        .recent-table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: auto;
            font-size: 0.86rem;
            line-height: 1.35;
        }}
        .recent-table th {{
            position: sticky;
            top: 0;
            z-index: 1;
            background: #f9fafb;
            border-bottom: 1px solid #e5e7eb;
            color: #374151;
            font-weight: 650;
            padding: 0.55rem 0.6rem;
            text-align: left;
            white-space: nowrap;
        }}
        .recent-table td {{
            min-height: 96px;
            border-bottom: 1px solid #eef2f7;
            color: #111827;
            padding: 0.65rem 0.6rem;
            vertical-align: top;
        }}
        .recent-cell-text {{
            min-height: 5.4em;
            max-height: 5.4em;
            overflow: hidden;
            white-space: normal;
            overflow-wrap: anywhere;
        }}
        </style>
        <div class="recent-table-wrap">
            <table class="recent-table">
                <thead><tr>{headers}</tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """
    ).strip()


def _render_admin_controls() -> None:
    with st.expander("Admin controls"):
        st.caption("Clearing logs requires MONITORING_ADMIN_PASSWORD. Without it, clear logs directly in Supabase with `truncate table public.rag_interactions restart identity;`.")
        expected_password = monitoring_admin_password()
        password = st.text_input("Admin password", type="password")
        disabled = not expected_password
        if st.button("Clear monitoring logs", type="secondary", disabled=disabled):
            if password == expected_password and clear_monitoring_logs():
                st.success("Monitoring logs cleared.")
                st.rerun()
            else:
                st.error("Could not clear logs. Check the password and Supabase permissions.")


def _format_metric_seconds(value) -> str:
    number = _num(value)
    return "N/A" if number is None else f"{number:.2f}s"


def _format_metric_number(value, decimals: int = 1) -> str:
    number = _num(value)
    return "N/A" if number is None else f"{number:.{decimals}f}"


def _format_percent(value) -> str:
    number = _num(value)
    return "N/A" if number is None else f"{number * 100:.1f}%"


def _truncate_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else f"{value[: limit - 3]}..."


def _render_mermaid(diagram: str, height: int = 720) -> None:
    escaped_diagram = html.escape(diagram)
    components.html(
        f"""
        <div class="mermaid">
        {escaped_diagram}
        </div>
        <script type="module">
            import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
            mermaid.initialize({{ startOnLoad: true, theme: "default" }});
        </script>
        """,
        height=height,
        scrolling=True,
    )


def render_about() -> None:
    settings = get_settings()
    st.subheader("About the App")
    st.write(
        "This portfolio app answers Autodesk-related questions with an evidence-grounded RAG workflow. "
        "It combines local Autodesk corpus retrieval, optional web evidence, reranking, strict evidence adequacy checks, "
        "LangSmith evaluation, and Supabase-backed runtime monitoring."
    )
    st.write(
        "The app is designed to be explored like a small production prototype. A user can ask a question, switch retrieval "
        "policies, compare evaluation results, and inspect runtime monitoring without needing to read the code. The main "
        "design goal is not just to produce an answer, but to make the answer path inspectable: what evidence was used, "
        "whether web search participated, whether the evidence was strong enough, how long the pipeline took, and where "
        "the app would refuse to answer."
    )

    st.subheader("Runtime Flow")
    st.write(
        "A user question enters the Ask tab with one of three search modes selected. All modes use the same local "
        "Docling + Chroma + BM25 hybrid retrieval backbone. Option 1 stays local-only, Option 2 adds official Autodesk.com "
        "web evidence, and Option 3 adds capped open-web evidence. Local chunks and web snippets are reranked together, "
        "then a strict adequacy gate checks whether the supplied evidence explicitly supports the requested answer. "
        "If the evidence is sufficient, the answer model generates a short sourced response; otherwise the app returns "
        "the fixed no-answer response."
    )
    st.write(
        "In everyday use, the flow starts in **Ask**. The user chooses a retrieval policy in **Settings & Eval**, returns "
        "to **Ask**, and submits a natural-language Autodesk question. The answer card then shows the generated answer, "
        "the selected search mode, the answer source type, routing notes, web-search details when relevant, and expandable "
        "sources. This makes it easier to tell whether a response came from the local corpus, official Autodesk web evidence, "
        "or broader open-web evidence."
    )
    st.table(
        [
            {"Stage": "Question intake", "What happens": "Streamlit captures the user question and active search mode."},
            {"Stage": "Local retrieval", "What happens": f"Chroma semantic search and BM25 keyword search retrieve candidates, then weighted RRF combines them with {settings.hybrid_vector_weight:.2f} vector / {settings.hybrid_bm25_weight:.2f} BM25 weighting."},
            {"Stage": "Context expansion", "What happens": "Neighbor chunks from the same source document are added within the context budget to reduce chunk-boundary misses."},
            {"Stage": "Optional web evidence", "What happens": "Option 2 searches Autodesk.com; Option 3 searches the open web with a smaller result cap."},
            {"Stage": "Reranking", "What happens": "A cross-encoder reranks local and web evidence blocks before answerability checking."},
            {"Stage": "Adequacy gate", "What happens": "The app verifies that the evidence explicitly supports the needed fact and refuses unsupported answers."},
            {"Stage": "Generation and logging", "What happens": "The final answer is generated from supplied evidence only, then interaction metadata is logged to Supabase when configured."},
        ]
    )

    st.subheader("Flowchart")
    st.write(
        "The flowchart below is the reviewer-friendly version of the runtime pipeline. It shows the three search modes "
        "branching early, then converging around the same evidence quality-control layer. The key idea is that web search "
        "does not bypass the RAG discipline: web snippets still become evidence blocks, compete with local chunks in the "
        "reranker, and must pass the same adequacy gate before generation."
    )
    _render_mermaid(
        """
flowchart TD
    A["User asks Autodesk question"] --> B["Streamlit app"]
    B --> C["Selected search mode"]

    C --> D1["Option 1: Local only"]
    C --> D2["Option 2: Local + Autodesk.com"]
    C --> D3["Option 3: Local + open web"]

    D1 --> E["Local hybrid retrieval"]
    D2 --> E
    D3 --> E

    E --> F1["Chroma semantic search"]
    E --> F2["BM25 keyword search"]
    F1 --> G["Weighted RRF fusion"]
    F2 --> G
    G --> H["Neighbor chunk expansion"]

    D2 --> W1["Autodesk.com web evidence"]
    D3 --> W2["Capped open-web evidence"]

    H --> R["Cross-encoder reranker"]
    W1 --> R
    W2 --> R

    R --> Q["Strict adequacy gate"]
    Q --> Z{"Explicitly supported?"}
    Z -->|Yes| Y["Generate grounded answer"]
    Z -->|No| N["Fixed no-answer response"]
    Y --> O["Display answer and sources"]
    N --> O
        """.strip()
    )

    st.subheader("Evaluation Metrics as of May 21, 2026")
    st.write(
        "These static values come from the saved 50-question golden-set evaluation runs completed on May 21, 2026. "
        "They are intentionally fixed for this About tab and will not change when a new evaluation is run."
    )
    st.write(
        "For active experimentation, the **Settings & Eval** tab can launch a fresh LangSmith evaluation for whichever "
        "option is selected. This table is different: it is a frozen portfolio snapshot so a reviewer always sees the same "
        "baseline comparison, even if later runs produce new results. The metrics summarize answer faithfulness, answer "
        "relevance, context quality, and latency across the same 50-question golden dataset."
    )
    st.table(
        [
            {
                "Search Option": "Option 1: Local Document Search",
                "Faithfulness": "0.91",
                "Answer Relevance": "0.655",
                "Context Precision": "0.72",
                "Context Recall": "0.575",
                "Avg Latency": "6.312s",
                "P50 Latency": "6.551s",
                "P99 Latency": "15.109s",
            },
            {
                "Search Option": "Option 2: Local + Autodesk.com",
                "Faithfulness": "0.87",
                "Answer Relevance": "0.85",
                "Context Precision": "0.805",
                "Context Recall": "0.735",
                "Avg Latency": "12.988s",
                "P50 Latency": "12.425s",
                "P99 Latency": "22.869s",
            },
            {
                "Search Option": "Option 3: Local + Open Web",
                "Faithfulness": "0.84",
                "Answer Relevance": "0.78",
                "Context Precision": "0.86",
                "Context Recall": "0.615",
                "Avg Latency": "14.112s",
                "P50 Latency": "12.335s",
                "P99 Latency": "73.169s",
            },
        ]
    )

    st.subheader("Current App Surfaces")
    st.write(
        "The app is organized around four tabs. **Ask** is the user-facing question-and-answer experience. "
        "**Settings & Eval** is the control room for changing search policy and running evaluation. **Monitoring** is the "
        "runtime observability dashboard backed by Supabase logs. **About the App** is the static tour you are reading now, "
        "intended to help reviewers understand the architecture without having to reconstruct it from notebooks or source files."
    )
    st.table(
        [
            {"Tab": "Ask", "Purpose": "Run the selected RAG mode and inspect answers, routing notes, web attempts, and sources."},
            {"Tab": "Settings & Eval", "Purpose": "Select the retrieval/web policy and run the fixed LangSmith evaluation workflow."},
            {"Tab": "Monitoring", "Purpose": "Review Supabase runtime logs, latency diagnostics, source usage, backend usage, web usage, recent interactions, and errors."},
            {"Tab": "About the App", "Purpose": "Provide a static reviewer-friendly overview of the architecture, flow logic, and frozen evaluation snapshot."},
        ]
    )
    st.write(
        "The Monitoring tab is especially useful after a few synthetic or real test questions. It records full questions and "
        "generated answers for this proof-of-concept, plus retrieval metadata, source summaries, latency breakdowns, token/cost "
        "metadata, no-answer outcomes, and errors. That lets the app demonstrate not only RAG behavior, but the kind of "
        "runtime observability expected when a RAG workflow is moved beyond a notebook."
    )
    st.info("Portfolio demonstration only. Verify product, pricing, system requirement, and release information directly with Autodesk.")


_init_state()
page = st.radio(
    "Navigation",
    PAGE_OPTIONS,
    horizontal=True,
    label_visibility="collapsed",
    key="selected_page",
    on_change=on_page_change,
)
if page == "Ask":
    render_ask()
elif page == "Settings & Eval":
    render_settings_eval()
elif page == "Monitoring":
    render_monitoring()
else:
    render_about()
