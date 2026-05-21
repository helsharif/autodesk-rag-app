"""Agentic RAG orchestration for Autodesk grounded answers."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date

from langchain_core.prompts import ChatPromptTemplate

from src.config import AUTODESK_WEB_MODE, HYBRID_BACKEND_NAME, LOCAL_ONLY_MODE, OPEN_WEB_MODE, get_chat_model, get_settings
from src.context_expansion import expand_retrieved_docs
from src.reranker import rerank_documents
from src.retriever import RetrievedSource, has_sufficient_retrieval, search_documents
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
                ("system", "You are a strict evidence sufficiency checker. Use only supplied evidence. Do not answer. Return valid JSON with answerable, required_fact, supporting_quote, source_id. Set answerable=true only when the exact fact needed by the question appears in the evidence. Numeric, date, price, version, compatibility, product capability, or procedural questions require the exact value or requirement. Related but incomplete evidence is not enough."),
                ("human", "Question:\n{question}\n\nEvidence:\n{evidence}\n\nIs the exact answer present?"),
            ]
        )
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"You answer questions about Autodesk and Autodesk products using only the supplied local excerpts, web results, and runtime context. Every factual claim must be supported by the supplied context. Do not use memory, prior turns, outside knowledge, assumptions, or likely values. Keep answers to 2-3 short paragraphs. Cite source names, local source IDs, or URLs inline when available. If evidence is insufficient, output exactly: {NO_ANSWER}"),
                ("human", "Runtime context:\nCurrent date: {current_date}\n\nQuestion:\n{question}\n\nLocal document evidence:\n{local_context}\n\nWeb evidence:\n{web_context}\n\nGrounded answer:"),
            ]
        )

    def answer(self, question: str, force_web: bool = False) -> AgentResult:
        route = self._route_query(question)
        if route.abstain:
            return AgentResult(NO_ANSWER, [], False, False, route_reason=route.reason, route_needs_web=route.needs_web)

        docs, sources = search_documents(question, collection_name=self.collection_name)
        docs, sources = expand_retrieved_docs(docs, sources, collection_name=self.collection_name)
        local_context = self._format_local_context(docs, sources)
        local_ok = has_sufficient_retrieval(sources)
        local_answerable = self._evidence_is_answerable(question, local_context) if local_ok else False
        if local_answerable:
            docs, sources = rerank_documents(question, docs, sources)
            local_context = self._format_local_context(docs, sources)
        web_allowed = self.search_mode in {AUTODESK_WEB_MODE, OPEN_WEB_MODE}
        use_web = web_allowed or force_web
        web_context = ""
        web_error = ""
        web_query = ""
        if use_web:
            web_query = self._web_query(question)
            try:
                web_context = web_search(web_query, max_results=self._web_result_limit())
            except Exception as exc:
                web_error = str(exc)
                logger.warning("Web search failed: %s", exc)
        has_web = bool(web_context.strip()) and "No web search results found" not in web_context
        web_answerable = self._evidence_is_answerable(question, web_context) if has_web else False
        web_primary = web_answerable and self._needs_web(question)

        evidence_parts: list[str] = []
        if web_primary:
            evidence_parts.append(web_context)
        else:
            if local_answerable:
                evidence_parts.append(local_context)
            if web_answerable:
                evidence_parts.append(web_context)
        evidence = "\n\n".join(evidence_parts)
        if self._is_current_date_question(question):
            evidence = f"Runtime current date: {date.today().strftime('%A, %B %d, %Y')}\n\n{evidence}"

        final_answerable = bool(evidence.strip()) and (
            web_primary
            or (local_answerable and not web_answerable)
            or (web_answerable and not local_answerable)
            or self._evidence_is_answerable(question, evidence)
        )
        result_contexts = self._result_contexts(docs, web_context if has_web else "")
        if not final_answerable:
            return AgentResult(NO_ANSWER, [], False, False, result_contexts, route.reason, route.needs_web, use_web, web_error, web_query)

        response = (self.answer_prompt | self.llm).invoke(
            {
                "current_date": date.today().strftime("%A, %B %d, %Y"),
                "question": question,
                "local_context": local_context if local_answerable else "No sufficient local evidence available.",
                "web_context": web_context if has_web else "No web evidence used.",
            }
        )
        answer = getattr(response, "content", str(response)).strip() or NO_ANSWER
        source_labels = []
        if has_web:
            source_labels.extend(self._web_source_labels(web_context))
        if local_answerable:
            source_labels.extend(self._source_labels(sources))
        if self._is_current_date_question(question):
            source_labels.append("Runtime context: current system date")
        return AgentResult(answer, source_labels[:8], local_answerable, has_web, result_contexts, route.reason, route.needs_web, use_web, web_error, web_query)

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
    def _web_source_labels(web_context: str) -> list[str]:
        return [line.strip() for line in web_context.splitlines() if line.startswith("http")]

    @staticmethod
    def _result_contexts(docs: list, web_context: str = "") -> list[str]:
        contexts = [doc.page_content for doc in docs]
        if web_context.strip():
            contexts.append(web_context)
        return contexts
