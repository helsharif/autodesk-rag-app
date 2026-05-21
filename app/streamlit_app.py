"""Streamlit frontend for the Autodesk Agentic RAG app."""

from __future__ import annotations

import csv
import html
import json
import os
import subprocess
import sys
import textwrap
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

PAGE_OPTIONS = ["Ask", "Settings & Eval", "About the App"]
EVAL_AUTO_REFRESH_SECONDS = 20


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


def sync_query_state(page: str | None = None, mode_label: str | None = None) -> None:
    if page and st.query_params.get("page") != page:
        st.query_params["page"] = page
    if mode_label and st.query_params.get("mode") != mode_label:
        st.query_params["mode"] = mode_label


def on_page_change() -> None:
    selected_page = st.session_state.get("selected_page", "Ask")
    if selected_page not in PAGE_OPTIONS:
        selected_page = "Ask"
    sync_query_state(page=selected_page, mode_label=st.session_state.get("search_mode_label", OPTION_1_LABEL))


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
    mode_label = st.session_state.get("search_mode_label", OPTION_1_LABEL)
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
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Routing, retrieving, checking evidence..."):
                result = _agent(st.session_state.collection_name, st.session_state.search_mode).answer(question)
            source_mode = _source_mode_label(result.used_local, result.used_web, st.session_state.search_mode)
            st.session_state.messages.append({"role": "assistant", "content": result.answer or NO_ANSWER, "sources": result.sources, "source_mode": source_mode, "route_reason": result.route_reason, "web_search_attempted": result.web_search_attempted, "web_query": result.web_query, "web_search_error": result.web_search_error, "search_mode_label": mode_label})
        except Exception as exc:
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


def render_settings_eval() -> None:
    settings = get_settings()
    st.subheader("Settings & Eval")
    labels = list(SEARCH_MODE_OPTIONS)
    current_label = st.session_state.get("search_mode_label", OPTION_1_LABEL)
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
        return float(value)
    except Exception:
        return None


def render_about() -> None:
    settings = get_settings()
    st.subheader("About the App")
    st.write("This portfolio app answers Autodesk-related questions with an agentic RAG workflow: local hybrid retrieval searches Chroma and BM25, deterministic neighbor expansion reduces chunk-boundary misses, and a strict adequacy gate refuses unsupported answers.")
    st.write("The Settings & Eval tab controls the web policy. Option 1 is local-only. Option 2 always adds autodesk.com web evidence. Option 3 always adds open web evidence. The final answer model receives only supplied excerpts, web snippets when enabled, and runtime date context.")
    st.table([
        {"Layer": "Retrieval", "Implementation": "Chroma dense semantic search + BM25 keyword search + Reciprocal Rank Fusion"},
        {"Layer": "Context", "Implementation": "Previous/current/next same-document chunk expansion with budget limits"},
        {"Layer": "Reranking", "Implementation": "SentenceTransformers CrossEncoder `cross-encoder/ms-marco-MiniLM-L6-v2` after the local adequacy gate"},
        {"Layer": "Generation", "Implementation": f"OpenAI `{settings.openai_model}` with strict source-grounded prompt"},
        {"Layer": "Evaluation", "Implementation": f"50-question golden dataset with five-point LLM judge scoring using `{settings.eval_judge_model}`"},
        {"Layer": "Web policy", "Implementation": "Autodesk.com mode uses up to 5 web results; open-web mode uses up to 3."},
    ])
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
else:
    render_about()
