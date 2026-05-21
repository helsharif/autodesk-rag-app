"""Background evaluation helpers for the Autodesk RAG app."""

from __future__ import annotations

import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langchain_core.prompts import ChatPromptTemplate

from src.agent import AutodeskRAGAgent, NO_ANSWER
from src.config import HYBRID_BACKEND_NAME, get_chat_model, get_settings


def run_evaluation(
    collection_name: str = HYBRID_BACKEND_NAME,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    rows = _load_testset(settings.eval_testset_path)
    agent = AutodeskRAGAgent(collection_name=collection_name)
    judge = get_chat_model(settings, temperature=0.0, model=settings.eval_judge_model)
    latencies: list[float] = []
    scored_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        question = row["question"]
        ground_truth = row["ground_truth"]
        if progress_callback:
            progress_callback({"phase": "answering", "message": "Running RAG over golden dataset.", "current": index - 1, "total": len(rows), "question": question})
        started = time.perf_counter()
        result = agent.answer(question)
        latency = time.perf_counter() - started
        latencies.append(latency)
        context = "\n\n".join(result.contexts or [])
        scores = _judge_scores(judge, question, ground_truth, result.answer, context)
        scored_rows.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "answer": result.answer,
                "latency": round(latency, 3),
                **scores,
            }
        )
        if progress_callback:
            progress_callback({"phase": "scoring", "message": "Scoring answer quality.", "current": index, "total": len(rows), "question": question})
        if settings.eval_judge_delay_seconds > 0:
            time.sleep(settings.eval_judge_delay_seconds)

    metrics = _aggregate(scored_rows, latencies)
    result_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_backend": collection_name,
        "question_count": len(rows),
        "metrics": metrics,
        "rows": scored_rows,
        "dataset_name": str(settings.eval_testset_path.relative_to(settings.root_dir)),
    }
    save_results(result_payload, collection_name)
    append_eval_results_log(result_payload)
    return result_payload


def save_results(payload: dict[str, Any], collection_name: str) -> Path:
    settings = get_settings()
    settings.eval_results_dir.mkdir(parents=True, exist_ok=True)
    path = settings.eval_results_dir / "docling_chroma_bm25_hybrid_results.json"
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
        "question_count": payload.get("question_count"),
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
        return [{"question": row["question"], "ground_truth": row["ground_truth"]} for row in reader]


def _judge_scores(judge, question: str, ground_truth: str, answer: str, context: str) -> dict[str, float]:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict RAG evaluator. Score each metric on exactly one of: 0.00, 0.25, 0.50, 0.75, 1.00. "
                "Faithfulness checks answer claims against context only; a conservative no-answer is faithful when context lacks the exact fact, but penalized when context clearly contains it. "
                "Answer relevance measures directness to the question. Context precision measures signal-to-noise in context. Context recall measures whether context contains facts needed for the reference answer. Return only JSON.",
            ),
            (
                "human",
                "Question:\n{question}\n\nReference answer:\n{ground_truth}\n\nRAG answer:\n{answer}\n\nRetrieved context:\n{context}\n\nReturn JSON with faithfulness, answer_relevancy, context_precision, context_recall.",
            ),
        ]
    )
    try:
        response = (prompt | judge).invoke({"question": question, "ground_truth": ground_truth, "answer": answer, "context": context[:18000]})
        text = getattr(response, "content", str(response)).strip()
        start = text.find("{")
        end = text.rfind("}")
        data = json.loads(text[start : end + 1] if start >= 0 and end >= start else text)
        return {key: _bucket(data.get(key)) for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
    except Exception:
        no_answer = answer.strip() == NO_ANSWER
        return {
            "faithfulness": 1.0 if no_answer else 0.5,
            "answer_relevancy": 0.25 if no_answer else 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
        }


def _bucket(value: Any) -> float:
    allowed = [0.0, 0.25, 0.5, 0.75, 1.0]
    try:
        score = float(value)
    except Exception:
        return 0.0
    return min(allowed, key=lambda candidate: abs(candidate - score))


def _aggregate(rows: list[dict[str, Any]], latencies: list[float]) -> dict[str, float]:
    metrics = {}
    for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        metrics[key] = round(statistics.mean(float(row[key]) for row in rows), 3) if rows else 0.0
    if latencies:
        sorted_latencies = sorted(latencies)
        metrics["average_latency"] = round(statistics.mean(latencies), 3)
        metrics["p50_latency"] = round(statistics.median(sorted_latencies), 3)
        metrics["p99_latency"] = round(sorted_latencies[min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.99))], 3)
    else:
        metrics.update({"average_latency": 0.0, "p50_latency": 0.0, "p99_latency": 0.0})
    return metrics
