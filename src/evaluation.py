"""LangSmith evaluation helpers for the Autodesk RAG app."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langchain_openai import ChatOpenAI
from langsmith import Client, traceable
from pydantic import BaseModel, Field

from src.agent import AutodeskRAGAgent, NO_ANSWER
from src.config import HYBRID_BACKEND_NAME, LIGHTRAG_AUTODESK_WEB_MODE, LOCAL_ONLY_MODE, get_settings


logger = logging.getLogger(__name__)
ProgressCallback = Callable[[dict[str, Any]], None]
EVAL_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
LANGSMITH_DATASET_PREFIX = "autodesk-rag-eval-testset"


class ScoredGrade(BaseModel):
    reasoning: str = Field(..., description="Brief scoring rationale.")
    score: float = Field(..., ge=0.0, le=1.0, description="One of 0.0, 0.25, 0.5, 0.75, or 1.0.")


def run_evaluation(
    collection_name: str = HYBRID_BACKEND_NAME,
    search_mode: str = LOCAL_ONLY_MODE,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper used by the background runner."""

    return run_langsmith_evaluation(collection_name, search_mode, progress_callback)


def run_langsmith_evaluation(
    collection_name: str = HYBRID_BACKEND_NAME,
    search_mode: str = LOCAL_ONLY_MODE,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY or OPEN_API_KEY is required to run evaluation.")
    if not settings.langsmith_api_key:
        raise ValueError("LANGSMITH_API_KEY is required to create LangSmith datasets and experiments.")

    _emit_progress(progress_callback, phase="loading_testset", message="Loading fixed 50-question CSV test set.", current=0, total=50)
    rows = _load_testset(settings.eval_testset_path)
    client = Client()
    dataset_name, dataset_id = _ensure_langsmith_dataset(client, rows, settings.eval_testset_path)
    ordered_examples = _ordered_langsmith_examples(client, dataset_id)

    _emit_progress(
        progress_callback,
        phase="dataset_ready",
        message=f"Using LangSmith dataset {dataset_name}.",
        current=0,
        total=len(rows),
    )

    started_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    latencies: list[dict[str, Any]] = []
    answer_counter = {"count": 0}

    @traceable(name=f"autodesk_rag_{search_mode}_target")
    def target(inputs: dict) -> dict:
        question = str(inputs["question"])
        answer_counter["count"] += 1
        current = answer_counter["count"]
        _emit_progress(
            progress_callback,
            phase="answering_questions",
            message=f"Running {search_mode} on question {current} of {len(rows)}.",
            current=current - 1,
            total=len(rows),
            question=question,
        )
        started = time.perf_counter()
        agent = AutodeskRAGAgent(collection_name=collection_name, search_mode=search_mode)
        error = None
        try:
            result = agent.answer(question)
            answer = result.answer
            contexts = list(result.contexts or [])
            sources = result.sources
            used_local = result.used_local
            used_web = result.used_web
        except Exception as exc:
            logger.exception("Evaluation target failed for question: %s", question)
            error = f"{exc.__class__.__name__}: {exc}"
            answer = NO_ANSWER
            contexts = []
            sources = []
            used_local = False
            used_web = False
        latency = round(time.perf_counter() - started, 3)
        latencies.append({"question": question, "execution_time": latency})
        _emit_progress(
            progress_callback,
            phase="answering_questions",
            message=f"Completed question {current} of {len(rows)} in {latency:.1f} seconds.",
            current=current,
            total=len(rows),
            question=question,
            execution_time=latency,
        )
        return {
            "answer": answer,
            "contexts": contexts,
            "sources": sources,
            "used_local": used_local,
            "used_web": used_web,
            "execution_time": latency,
            "error": error,
        }

    _emit_progress(
        progress_callback,
        phase="langsmith_evaluation",
        message="Running LangSmith experiment and LLM-as-judge evaluators.",
        current=0,
        total=len(rows),
    )
    experiment_results = client.evaluate(
        target,
        data=ordered_examples,
        evaluators=_build_langsmith_evaluators(),
        experiment_prefix=f"autodesk-rag-{search_mode}",
        description=f"Autodesk RAG evaluation for search mode {search_mode}.",
        metadata={
            "collection_name": collection_name,
            "search_mode": search_mode,
            "testset_path": str(settings.eval_testset_path.relative_to(settings.root_dir)),
            "testset_sha256": _testset_hash(rows),
        },
        max_concurrency=2,
        blocking=True,
        upload_results=True,
    )

    _emit_progress(
        progress_callback,
        phase="retrieving_scores",
        message="Retrieving LangSmith scores for dashboard display.",
        current=len(rows),
        total=len(rows),
    )
    scores_df = experiment_results.to_pandas()
    metrics = _extract_metric_means(scores_df)
    execution_times = _extract_execution_time_records(scores_df, latencies)
    metrics.update(_latency_metrics(execution_times))

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "started_at_utc": started_at_utc,
        "evaluation_backend": "LangSmith",
        "collection_name": collection_name,
        "search_mode": search_mode,
        "question_count": len(rows),
        "testset_path": str(settings.eval_testset_path.relative_to(settings.root_dir)),
        "testset_sha256": _testset_hash(rows),
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "experiment_name": experiment_results.experiment_name,
        "experiment_id": str(experiment_results.experiment_id),
        "experiment_url": experiment_results.url,
        "metrics": metrics,
        "execution_times": execution_times,
        "rows": _json_safe_records(scores_df),
    }
    save_results(payload, collection_name, search_mode)
    append_eval_results_log(payload)
    return payload


def save_results(payload: dict[str, Any], collection_name: str, search_mode: str = LOCAL_ONLY_MODE) -> Path:
    settings = get_settings()
    settings.eval_results_dir.mkdir(parents=True, exist_ok=True)
    path = settings.eval_results_dir / _result_filename(search_mode)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def append_eval_results_log(payload: dict[str, Any]) -> None:
    settings = get_settings()
    settings.eval_results_dir.mkdir(parents=True, exist_ok=True)
    path = settings.eval_results_dir / "eval_results_log.csv"
    metrics = payload.get("metrics", {})
    row = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "evaluation_backend": payload.get("evaluation_backend"),
        "search_mode": payload.get("search_mode"),
        "question_count": payload.get("question_count"),
        "experiment_url": payload.get("experiment_url"),
        "faithfulness": metrics.get("faithfulness"),
        "answer_relevancy": metrics.get("answer_relevancy"),
        "context_precision": metrics.get("context_precision"),
        "context_recall": metrics.get("context_recall"),
        "average_latency": metrics.get("average_latency"),
        "p50_latency": metrics.get("p50_latency"),
        "p99_latency": metrics.get("p99_latency"),
    }
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _load_testset(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = [{"question": row["question"].strip(), "ground_truth": row["ground_truth"].strip()} for row in reader]
    rows = [row for row in rows if row["question"] and row["ground_truth"]]
    if len(rows) != 50:
        raise ValueError(f"Evaluation CSV must contain exactly 50 populated rows; found {len(rows)}.")
    return rows


def _ensure_langsmith_dataset(client: Client, rows: list[dict[str, str]], testset_path: Path) -> tuple[str, str]:
    settings = get_settings()
    dataset_hash = _testset_hash(rows)
    dataset_name = f"{LANGSMITH_DATASET_PREFIX}-{dataset_hash}"
    examples = [
        {
            "inputs": {"question": row["question"]},
            "outputs": {"answer": row["ground_truth"]},
            "metadata": {"source": str(testset_path.relative_to(settings.root_dir)), "row_index": index},
        }
        for index, row in enumerate(rows)
    ]
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        existing_count = sum(1 for _ in client.list_examples(dataset_id=dataset.id))
        if existing_count == len(examples):
            return dataset_name, str(dataset.id)
        dataset_name = f"{dataset_name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Fixed 50-question Autodesk RAG evaluation set loaded from eval_testset/autodesk_testset.csv.",
        metadata={"testset_sha256": dataset_hash, "row_count": len(examples)},
    )
    client.create_examples(dataset_id=dataset.id, examples=examples)
    return dataset_name, str(dataset.id)


def _ordered_langsmith_examples(client: Client, dataset_id: str) -> list[Any]:
    """Return LangSmith examples sorted to match eval_testset/autodesk_testset.csv."""

    examples = list(client.list_examples(dataset_id=dataset_id))

    def row_index(example: Any) -> int:
        metadata = getattr(example, "metadata", None) or {}
        try:
            return int(metadata.get("row_index", 10**9))
        except Exception:
            return 10**9

    return sorted(examples, key=row_index)


def _build_langsmith_evaluators() -> list[Callable]:
    settings = get_settings()
    grader = ChatOpenAI(model=settings.eval_judge_model, temperature=0)
    structured = grader.with_structured_output(ScoredGrade, method="json_schema", strict=True)
    score_scale = "Use only: 0.00, 0.25, 0.50, 0.75, 1.00."

    def faithfulness(inputs: dict, outputs: dict) -> dict:
        grade = _judge(
            structured,
            "Grade faithfulness for an Autodesk RAG answer. Use only QUESTION, ANSWER, and CONTEXT. "
            "Every factual claim must be supported by supplied context. Conservative no-answer is faithful only when context lacks the exact fact. "
            f"Be strict with versions, dates, prices, product names, system requirements, and procedures. {score_scale}",
            inputs,
            outputs,
        )
        return _feedback("faithfulness", grade)

    def answer_relevancy(inputs: dict, outputs: dict) -> dict:
        grade = _judge(
            structured,
            "Grade answer relevance for an Autodesk RAG answer. Score how directly the answer addresses the user question. "
            f"Penalize irrelevant, incomplete, or unnecessary answers. {score_scale}",
            inputs,
            outputs,
        )
        return _feedback("answer_relevancy", grade)

    def context_precision(inputs: dict, outputs: dict) -> dict:
        grade = _judge(
            structured,
            "Grade context precision for Autodesk RAG retrieval. Score signal-to-noise in CONTEXT for the QUESTION. "
            f"Web snippets are valid context when supplied, but noisy third-party snippets should reduce precision. {score_scale}",
            inputs,
            outputs,
        )
        return _feedback("context_precision", grade)

    def context_recall(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        grade = _judge(
            structured,
            "Grade context recall for Autodesk RAG retrieval. Compare CONTEXT to the REFERENCE ANSWER and score whether required facts are present. "
            f"Web snippets are valid retrieved context when supplied. Be strict with versions, dates, product names, prices, and requirements. {score_scale}",
            inputs,
            outputs,
            reference_outputs,
        )
        return _feedback("context_recall", grade)

    return [faithfulness, answer_relevancy, context_precision, context_recall]


def _judge(llm, instructions: str, inputs: dict, outputs: dict, reference_outputs: dict | None = None):
    reference_text = ""
    if reference_outputs:
        reference_text = f"\n\nREFERENCE ANSWER:\n{reference_outputs.get('answer', '')}"
    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{inputs.get('question', '')}\n\n"
                f"ANSWER:\n{outputs.get('answer', '')}{reference_text}\n\n"
                f"CONTEXT:\n{_contexts_to_text(outputs)}"
            ),
        },
    ]
    settings = get_settings()
    if settings.eval_judge_delay_seconds:
        time.sleep(settings.eval_judge_delay_seconds)
    return llm.invoke(messages)


def _feedback(key: str, grade: ScoredGrade | dict) -> dict:
    score = getattr(grade, "score", 0.0) if isinstance(grade, BaseModel) else grade.get("score", 0.0)
    reasoning = getattr(grade, "reasoning", "") if isinstance(grade, BaseModel) else grade.get("reasoning", "")
    return {"key": key, "score": _bucket(score), "comment": reasoning}


def _contexts_to_text(outputs: dict, limit: int = 30000) -> str:
    contexts = outputs.get("contexts") or []
    if not contexts:
        return "No retrieved context."
    blocks = []
    for index, context in enumerate(contexts, start=1):
        blocks.append(f"[Context {index}]\n{context}")
    return "\n\n".join(blocks)[:limit]


def _extract_metric_means(scores_df) -> dict[str, float | None]:
    metrics = {}
    for metric in EVAL_METRICS:
        column = f"feedback.{metric}"
        if column in scores_df.columns:
            metrics[metric] = _safe_mean(scores_df[column])
        elif metric in scores_df.columns:
            metrics[metric] = _safe_mean(scores_df[metric])
        else:
            metrics[metric] = None
    return metrics


def _extract_execution_time_records(scores_df, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if "execution_time" not in scores_df.columns:
        return fallback
    records = []
    for _, row in scores_df.iterrows():
        value = _safe_float(row.get("execution_time"))
        if value is not None:
            records.append({"question": str(row.get("inputs.question") or ""), "execution_time": round(value, 3)})
    return records or fallback


def _latency_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [float(row["execution_time"]) for row in records if row.get("execution_time") is not None]
    if not values:
        return {"average_latency": None, "p50_latency": None, "p99_latency": None}
    values = sorted(values)
    p99_index = min(len(values) - 1, int(len(values) * 0.99))
    return {
        "average_latency": round(statistics.mean(values), 3),
        "p50_latency": round(statistics.median(values), 3),
        "p99_latency": round(values[p99_index], 3),
    }


def _bucket(value: Any) -> float:
    allowed = [0.0, 0.25, 0.5, 0.75, 1.0]
    try:
        score = float(value)
    except Exception:
        return 0.0
    return min(allowed, key=lambda candidate: abs(candidate - score))


def _testset_hash(rows: list[dict[str, str]]) -> str:
    normalized = "\n".join(f"{row['question']}\t{row['ground_truth']}" for row in rows)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value != value:
            return None
        return float(value)
    except Exception:
        return None


def _safe_mean(series) -> float | None:
    values = [_safe_float(value) for value in series]
    values = [value for value in values if value is not None]
    return round(statistics.mean(values), 3) if values else None


def _json_safe_records(dataframe) -> list[dict[str, Any]]:
    records = dataframe.to_dict(orient="records")
    return [{key: _json_safe(value) for key, value in record.items()} for record in records]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _result_filename(search_mode: str) -> str:
    if search_mode == LIGHTRAG_AUTODESK_WEB_MODE:
        return "option_4_lightrag_mixed_autodesk_web_results.json"
    if search_mode == "autodesk_web":
        return "docling_chroma_bm25_hybrid_autodesk_web_results.json"
    if search_mode == "open_web":
        return "docling_chroma_bm25_hybrid_open_web_results.json"
    return "docling_chroma_bm25_hybrid_results.json"


def _emit_progress(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback:
        progress_callback(payload)
