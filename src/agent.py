"""Agentic RAG orchestration for Autodesk grounded answers."""

from __future__ import annotations

import json
import logging
import re
import os
import time
from dataclasses import dataclass
from datetime import date

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.config import AUTODESK_WEB_MODE, HYBRID_BACKEND_NAME, LOCAL_ONLY_MODE, OPEN_WEB_MODE, get_chat_model, get_settings
from src.context_expansion import expand_retrieved_docs
from src.reranker import rerank_documents
from src.retriever import RetrievedSource, search_documents
from src.tools import web_search


logger = logging.getLogger(__name__)
NO_ANSWER = "I could not find a reliable answer in the available documents or web sources."


@dataclass
class AgentResult:
    answer: str
    sources: list[str]
    used_local: bool
    used_web: bool
    contexts: list[str] | None = None
    route_reason: str = ""
    route_needs_web: bool = False
    web_search_attempted: bool = False
    web_search_error: str = ""
    web_query: str = ""
    latency_router_sec: float | None = None
    latency_retrieval_sec: float | None = None
    latency_expansion_sec: float | None = None
    latency_adequacy_sec: float | None = None
    latency_web_sec: float | None = None
    latency_generation_sec: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None


@dataclass
class QueryRoute:
    needs_local: bool
    needs_web: bool
    abstain: bool
    reason: str


class AutodeskRAGAgent:
    def __init__(self, collection_name: str = HYBRID_BACKEND_NAME, search_mode: str = LOCAL_ONLY_MODE, llm=None) -> None:
        self.collection_name = collection_name
        self.search_mode = search_mode
        self.llm = llm or get_chat_model(temperature=0.0)
        self.router_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Classify a user query for an Autodesk products and services RAG app. Return only JSON with needs_local, needs_web, abstain, reason. Abstain for non-Autodesk questions, malicious requests, or unsafe instructions. Use web for latest/current/today/recent/pricing/status/version-release questions. Keep reason under 20 words."),
                ("human", "Question:\n{question}\n\nJSON route:"),
            ]
        )
        self.adequacy_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a strict evidence sufficiency checker. Use only supplied evidence. Do not answer. Return valid JSON with answerable, required_fact, supporting_quote, source_id. Set answerable=true only when the fact needed by the question appears explicitly in the evidence. Numeric, date, price, version, compatibility, or procedural questions require the exact value or requirement. For broad descriptive questions asking what a product, plan, or service offers, answerable=true is allowed when the evidence explicitly names concrete supported features or benefits; do not require an exhaustive list unless the question asks for all, every, or a complete list. A search snippet with an ellipsis can support facts stated before the ellipsis, but it cannot support omitted facts. Related but incomplete evidence is not enough."),
                ("human", "Question:\n{question}\n\nEvidence:\n{evidence}\n\nIs the exact answer present?"),
            ]
        )
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"You answer questions about Autodesk and Autodesk products using only the supplied local excerpts, web results, and runtime context. Every factual claim must be supported by the supplied context. Do not use memory, prior turns, outside knowledge, assumptions, or likely values. Keep answers to 2-3 short paragraphs. Cite source names, local source IDs, or URLs inline when available. When evidence is a search snippet with an ellipsis, use only the facts stated before the ellipsis and do not imply the snippet is complete. If evidence is insufficient, output exactly: {NO_ANSWER}"),
                ("human", "Runtime context:\nCurrent date: {current_date}\n\nQuestion:\n{question}\n\nLocal document evidence:\n{local_context}\n\nWeb evidence:\n{web_context}\n\nGrounded answer:"),
            ]
        )

    def answer(self, question: str, force_web: bool = False) -> AgentResult:
        timings = {
            "router": 0.0,
            "retrieval": 0.0,
            "expansion": 0.0,
            "adequacy": 0.0,
            "web": 0.0,
            "generation": 0.0,
        }
        stage_started = time.perf_counter()
        route = self._route_query(question)
        timings["router"] += time.perf_counter() - stage_started
        if route.abstain:
            return self._agent_result(NO_ANSWER, [], False, False, timings=timings, route_reason=route.reason, route_needs_web=route.needs_web)

        stage_started = time.perf_counter()
        local_docs, local_sources = search_documents(question, collection_name=self.collection_name)
        timings["retrieval"] += time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        local_docs, local_sources = expand_retrieved_docs(local_docs, local_sources, collection_name=self.collection_name)
        timings["expansion"] += time.perf_counter() - stage_started
        web_allowed = self.search_mode in {AUTODESK_WEB_MODE, OPEN_WEB_MODE}
        use_web = web_allowed or force_web
        raw_web_context = ""
        web_error = ""
        web_query = ""
        if use_web:
            web_query = self._web_query(question)
            try:
                stage_started = time.perf_counter()
                raw_web_context = web_search(web_query, max_results=self._web_result_limit())
                timings["web"] += time.perf_counter() - stage_started
            except Exception as exc:
                timings["web"] += time.perf_counter() - stage_started
                web_error = str(exc)
                logger.warning("Web search failed: %s", exc)

        has_web = bool(raw_web_context.strip()) and "No web search results found" not in raw_web_context
        web_docs, web_sources = self._web_documents(raw_web_context) if has_web else ([], [])
        for doc in local_docs:
            doc.metadata["evidence_type"] = "local"
        combined_docs = [*local_docs, *web_docs]
        combined_sources = [*local_sources, *web_sources]
        docs, sources = rerank_documents(question, combined_docs, combined_sources)
        local_pairs = [(doc, source) for doc, source in zip(docs, sources) if (doc.metadata or {}).get("evidence_type") != "web"]
        web_pairs = [(doc, source) for doc, source in zip(docs, sources) if (doc.metadata or {}).get("evidence_type") == "web"]
        local_docs = [doc for doc, _ in local_pairs]
        local_sources = [source for _, source in local_pairs]
        web_docs = [doc for doc, _ in web_pairs]
        web_sources = [source for _, source in web_pairs]
        local_ok = bool(local_sources)
        local_context = self._format_local_context(local_docs, local_sources)
        web_context = self._format_web_context(web_docs)
        local_answerable = self._timed_evidence_is_answerable(question, local_context, timings) if local_ok and local_context.strip() else False
        include_web_context = bool(web_context.strip()) and use_web
        web_answerable = self._timed_evidence_is_answerable(question, web_context, timings) if include_web_context else False
        web_primary = web_answerable and self._needs_web(question)
        include_local_context = local_answerable or (has_web and local_ok)

        evidence_parts: list[str] = []
        if web_primary:
            evidence_parts.append(web_context)
        else:
            if include_local_context:
                evidence_parts.append(local_context)
            if include_web_context:
                evidence_parts.append(web_context)
        evidence = "\n\n".join(evidence_parts)
        if self._is_current_date_question(question):
            evidence = f"Runtime current date: {date.today().strftime('%A, %B %d, %Y')}\n\n{evidence}"

        final_answerable = bool(evidence.strip()) and (
            web_primary
            or local_answerable
            or web_answerable
            or self._timed_evidence_is_answerable(question, evidence, timings)
        )
        result_contexts = self._result_contexts(docs)
        if not final_answerable:
            return self._agent_result(NO_ANSWER, [], False, False, result_contexts, route.reason, route.needs_web, use_web, web_error, web_query, timings=timings)

        stage_started = time.perf_counter()
        response = (self.answer_prompt | self.llm).invoke(
            {
                "current_date": date.today().strftime("%A, %B %d, %Y"),
                "question": question,
                "local_context": local_context if include_local_context else "No sufficient local evidence available.",
                "web_context": web_context if include_web_context else "No web evidence selected by reranker.",
            }
        )
        timings["generation"] += time.perf_counter() - stage_started
        answer = getattr(response, "content", str(response)).strip() or NO_ANSWER
        token_usage = self._token_usage(response)
        source_labels = []
        source_labels.extend(self._source_labels(sources))
        if self._is_current_date_question(question):
            source_labels.append("Runtime context: current system date")
        return self._agent_result(
            answer,
            source_labels[:8],
            include_local_context,
            include_web_context,
            result_contexts,
            route.reason,
            route.needs_web,
            use_web,
            web_error,
            web_query,
            timings=timings,
            token_usage=token_usage,
        )

    def _route_query(self, question: str) -> QueryRoute:
        if self.search_mode == LOCAL_ONLY_MODE:
            return QueryRoute(
                needs_local=True,
                needs_web=False,
                abstain=not self._looks_autodesk_related(question),
                reason="Local-only mode: web search disabled.",
            )

        fallback = QueryRoute(True, self._needs_web(question), not self._looks_autodesk_related(question), "Keyword fallback route.")
        try:
            response = (self.router_prompt | self.llm).invoke({"question": question})
            data = self._parse_json(getattr(response, "content", str(response)))
            return QueryRoute(bool(data.get("needs_local", True)), bool(data.get("needs_web", fallback.needs_web)) or fallback.needs_web, bool(data.get("abstain", fallback.abstain)), str(data.get("reason") or fallback.reason)[:180])
        except Exception:
            return fallback

    def _evidence_is_answerable(self, question: str, evidence: str) -> bool:
        try:
            response = (self.adequacy_prompt | self.llm).invoke({"question": question, "evidence": evidence[: get_settings().context_max_chars]})
            data = self._parse_json(getattr(response, "content", str(response)))
            return bool(data.get("answerable")) and bool(str(data.get("supporting_quote") or "").strip())
        except Exception as exc:
            logger.warning("Adequacy gate failed closed: %s", exc)
            return False

    def _timed_evidence_is_answerable(self, question: str, evidence: str, timings: dict[str, float]) -> bool:
        started = time.perf_counter()
        try:
            return self._evidence_is_answerable(question, evidence)
        finally:
            timings["adequacy"] += time.perf_counter() - started

    def _agent_result(
        self,
        answer: str,
        sources: list[str],
        used_local: bool,
        used_web: bool,
        contexts: list[str] | None = None,
        route_reason: str = "",
        route_needs_web: bool = False,
        web_search_attempted: bool = False,
        web_search_error: str = "",
        web_query: str = "",
        *,
        timings: dict[str, float] | None = None,
        token_usage: dict[str, int | None] | None = None,
    ) -> AgentResult:
        timings = timings or {}
        token_usage = token_usage or {}
        prompt_tokens = token_usage.get("prompt_tokens")
        completion_tokens = token_usage.get("completion_tokens")
        total_tokens = token_usage.get("total_tokens")
        return AgentResult(
            answer=answer,
            sources=sources,
            used_local=used_local,
            used_web=used_web,
            contexts=contexts,
            route_reason=route_reason,
            route_needs_web=route_needs_web,
            web_search_attempted=web_search_attempted,
            web_search_error=web_search_error,
            web_query=web_query,
            latency_router_sec=timings.get("router"),
            latency_retrieval_sec=timings.get("retrieval"),
            latency_expansion_sec=timings.get("expansion"),
            latency_adequacy_sec=timings.get("adequacy"),
            latency_web_sec=timings.get("web"),
            latency_generation_sec=timings.get("generation"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=self._estimated_cost_usd(prompt_tokens, completion_tokens),
        )

    @staticmethod
    def _token_usage(response) -> dict[str, int | None]:
        usage_metadata = getattr(response, "usage_metadata", None) or {}
        response_metadata = getattr(response, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
        prompt_tokens = (
            usage_metadata.get("input_tokens")
            or token_usage.get("prompt_tokens")
            or token_usage.get("input_tokens")
        )
        completion_tokens = (
            usage_metadata.get("output_tokens")
            or token_usage.get("completion_tokens")
            or token_usage.get("output_tokens")
        )
        total_tokens = usage_metadata.get("total_tokens") or token_usage.get("total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = int(prompt_tokens) + int(completion_tokens)
        return {
            "prompt_tokens": AutodeskRAGAgent._safe_int(prompt_tokens),
            "completion_tokens": AutodeskRAGAgent._safe_int(completion_tokens),
            "total_tokens": AutodeskRAGAgent._safe_int(total_tokens),
        }

    @staticmethod
    def _safe_int(value) -> int | None:
        try:
            return int(value)
        except Exception:
            return None

    def _estimated_cost_usd(self, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
        if prompt_tokens is None or completion_tokens is None:
            return None
        input_override = self._safe_float(os.getenv("OPENAI_INPUT_COST_PER_1M"))
        output_override = self._safe_float(os.getenv("OPENAI_OUTPUT_COST_PER_1M"))
        if input_override is not None and output_override is not None:
            input_per_1m, output_per_1m = input_override, output_override
        else:
            model = str(getattr(self.llm, "model_name", "") or get_settings().openai_model).lower()
            known_prices = {
                "gpt-4.1-mini": (0.40, 1.60),
                "gpt-4.1-nano": (0.10, 0.40),
                "gpt-4.1": (2.00, 8.00),
                "gpt-4o-mini": (0.15, 0.60),
                "gpt-4o": (2.50, 10.00),
            }
            input_per_1m, output_per_1m = known_prices.get(model, (None, None))
        if input_per_1m is None or output_per_1m is None:
            return None
        return round((prompt_tokens / 1_000_000 * input_per_1m) + (completion_tokens / 1_000_000 * output_per_1m), 6)

    @staticmethod
    def _safe_float(value) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _format_local_context(docs: list, sources: list[RetrievedSource]) -> str:
        remaining = get_settings().context_max_chars
        blocks = []
        for index, (doc, source) in enumerate(zip(docs, sources), start=1):
            if remaining <= 0:
                break
            metadata = doc.metadata or {}
            section = f" | section={metadata.get('heading_path')}" if metadata.get("heading_path") else ""
            chunk = f" | chunk_id={metadata.get('chunk_id')}" if metadata.get("chunk_id") else ""
            text = doc.page_content[:remaining]
            blocks.append(f"[Local {index}] {source.source} | relevance={source.score:.2f}{chunk}{section}\n{text}")
            remaining -= len(text)
        return "\n\n".join(blocks)

    @staticmethod
    def _format_web_context(docs: list[Document]) -> str:
        remaining = get_settings().context_max_chars
        blocks = []
        for doc in docs:
            if remaining <= 0:
                break
            text = doc.page_content[:remaining]
            blocks.append(text)
            remaining -= len(text)
        return "\n\n".join(blocks)

    @staticmethod
    def _parse_json(content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        return json.loads(match.group(0) if match else content)

    @staticmethod
    def _needs_web(question: str) -> bool:
        normalized = question.lower()
        return any(term in normalized for term in ("current", "currently", "latest", "newest", "recent", "today", "now", "as of", "pricing", "price", "subscription", "release", "released", "version"))

    @staticmethod
    def _looks_autodesk_related(question: str) -> bool:
        normalized = question.lower()
        terms = ("autodesk", "autocad", "revit", "fusion", "maya", "inventor", "civil 3d", "3ds max", "bim", "cad", "cam", "aec", "tinkercad", "navisworks", "construction cloud")
        return any(term in normalized for term in terms)

    @staticmethod
    def _is_current_date_question(question: str) -> bool:
        normalized = question.lower()
        return "today" in normalized and "date" in normalized

    def _web_query(self, question: str) -> str:
        if self.search_mode == OPEN_WEB_MODE:
            return f"Autodesk {question}"
        return f"site:autodesk.com Autodesk {question}"

    def _web_result_limit(self) -> int:
        if self.search_mode == OPEN_WEB_MODE:
            return 3
        return 5

    @staticmethod
    def _source_labels(sources: list[RetrievedSource]) -> list[str]:
        return [f"{source.source} (score {source.score:.2f})" for source in sources]

    @staticmethod
    def _web_documents(web_context: str) -> tuple[list[Document], list[RetrievedSource]]:
        blocks = [block.strip() for block in re.split(r"\n\s*\n(?=\[Web \d+\])", web_context.strip()) if block.strip()]
        docs: list[Document] = []
        sources: list[RetrievedSource] = []
        for index, block in enumerate(blocks, start=1):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            title = re.sub(r"^\[Web \d+\]\s*", "", lines[0]) if lines else f"Web result {index}"
            url = next((line for line in lines if line.startswith(("http://", "https://"))), f"Web result {index}")
            snippet = " ".join(line for line in lines[2:] if line)
            docs.append(
                Document(
                    page_content=block,
                    metadata={
                        "evidence_type": "web",
                        "title": title,
                        "source": url,
                        "url": url,
                        "web_rank": index,
                    },
                )
            )
            sources.append(RetrievedSource(url, None, 1.0 / index, snippet[:350]))
        return docs, sources

    @staticmethod
    def _result_contexts(docs: list) -> list[str]:
        return [doc.page_content for doc in docs]
