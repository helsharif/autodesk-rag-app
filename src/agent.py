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


@dataclass
class CompareRetrievalPlan:
    is_compare: bool
    products: list[str]
    subqueries: list[str]


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
        self.compare_adequacy_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a strict evidence sufficiency checker for Autodesk compare/contrast questions. Use only supplied evidence. Do not answer the user. Return valid JSON with answerable, supported_entities, missing_entities, supporting_quotes, source_ids, direct_comparison_present. A direct comparison passage is helpful but not required. Set answerable=true when the evidence explicitly provides substantive facts about each compared entity, even if those facts appear in separate excerpts. Set answerable=false if any compared entity is only mentioned in passing or lacks concrete supported facts. Do not infer product capabilities, industries, or recommendations beyond the evidence."),
                ("human", "Question:\n{question}\n\nCompared entities:\n{entities}\n\nEvidence:\n{evidence}\n\nCan a grounded comparison be synthesized from this evidence?"),
            ]
        )
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"You answer questions about Autodesk and Autodesk products using only the supplied local excerpts, web results, and runtime context. Every factual claim must be supported by the supplied context. Do not use memory, prior turns, outside knowledge, assumptions, or likely values. For compare/contrast questions, you may synthesize a side-by-side comparison from separate evidence about each product; if the supplied evidence does not directly compare them, say that the comparison is synthesized from separate retrieved evidence. Do not rank products or recommend one unless the supplied evidence supports the use-case criteria. Keep answers to 2-3 short paragraphs. Cite source names, local source IDs, or URLs inline when available. When evidence is a search snippet with an ellipsis, use only the facts stated before the ellipsis and do not imply the snippet is complete. If evidence is insufficient, output exactly: {NO_ANSWER}"),
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
        local_docs, local_sources, compare_plan = self._retrieve_local_documents(question)
        timings["retrieval"] += time.perf_counter() - stage_started
        route_reason = route.reason
        if compare_plan.is_compare:
            route_reason = f"{route.reason} Compare/contrast retrieval for: {', '.join(compare_plan.products) or 'detected entities'}."
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
        if compare_plan.is_compare and len(compare_plan.products) >= 2:
            docs, sources = self._ensure_compare_balance_after_rerank(
                docs,
                sources,
                combined_docs,
                combined_sources,
                compare_plan.products,
                max(1, min(get_settings().reranker_top_n, len(combined_docs))),
            )
        local_pairs = [(doc, source) for doc, source in zip(docs, sources) if (doc.metadata or {}).get("evidence_type") != "web"]
        web_pairs = [(doc, source) for doc, source in zip(docs, sources) if (doc.metadata or {}).get("evidence_type") == "web"]
        local_docs = [doc for doc, _ in local_pairs]
        local_sources = [source for _, source in local_pairs]
        web_docs = [doc for doc, _ in web_pairs]
        web_sources = [source for _, source in web_pairs]
        local_ok = bool(local_sources)
        local_context = self._format_local_context(local_docs, local_sources)
        web_context = self._format_web_context(web_docs)
        compare_products = compare_plan.products if compare_plan.is_compare else None
        local_answerable = False
        if local_ok and local_context.strip():
            local_answerable = self._compare_evidence_has_entity_coverage(local_docs, local_sources, compare_products) if compare_products else self._timed_evidence_is_answerable(question, local_context, timings)
        include_web_context = bool(web_context.strip()) and use_web
        web_answerable = False
        if include_web_context:
            web_answerable = self._compare_evidence_has_entity_coverage(web_docs, web_sources, compare_products) if compare_products else self._timed_evidence_is_answerable(question, web_context, timings)
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
            return self._agent_result(NO_ANSWER, [], False, False, result_contexts, route_reason, route.needs_web, use_web, web_error, web_query, timings=timings)

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
            route_reason,
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

    def _retrieve_local_documents(self, question: str) -> tuple[list[Document], list[RetrievedSource], CompareRetrievalPlan]:
        compare_plan = self._compare_retrieval_plan(question)
        if not compare_plan.is_compare:
            docs, sources = search_documents(question, collection_name=self.collection_name)
            return docs, sources, compare_plan

        settings = get_settings()
        retrieval_query = self._expanded_compare_retrieval_query(question, compare_plan)
        retrieval_k = min(max(settings.retriever_k * 2, settings.retriever_k), settings.hybrid_candidate_k)
        logger.info(
            "Compare/contrast retrieval triggered. products=%s expanded_subqueries=%s",
            compare_plan.products,
            compare_plan.subqueries,
        )

        pairs: list[tuple[Document, RetrievedSource]] = []
        docs, sources = search_documents(retrieval_query, k=retrieval_k, collection_name=self.collection_name)
        for doc, source in zip(docs, sources):
            metadata = dict(doc.metadata or {})
            metadata["compare_retrieval_query"] = retrieval_query
            metadata["compare_retrieval_subqueries"] = " | ".join(compare_plan.subqueries)
            pairs.append((Document(page_content=doc.page_content, metadata=metadata), source))

        deduped_pairs = self._dedupe_document_pairs(pairs)
        balanced_pairs = self._select_balanced_compare_pairs(deduped_pairs, compare_plan.products, settings.retriever_k)
        return [doc for doc, _ in balanced_pairs], [source for _, source in balanced_pairs], compare_plan

    def _evidence_is_answerable(self, question: str, evidence: str, compare_products: list[str] | None = None) -> bool:
        try:
            if compare_products and len(compare_products) >= 2:
                response = (self.compare_adequacy_prompt | self.llm).invoke(
                    {
                        "question": question,
                        "entities": ", ".join(compare_products[:4]),
                        "evidence": evidence[: get_settings().context_max_chars],
                    }
                )
                data = self._parse_json(getattr(response, "content", str(response)))
                return (
                    bool(data.get("answerable"))
                    and bool(data.get("supporting_quotes"))
                    and self._compare_entities_supported(compare_products, data.get("supported_entities"))
                )
            response = (self.adequacy_prompt | self.llm).invoke({"question": question, "evidence": evidence[: get_settings().context_max_chars]})
            data = self._parse_json(getattr(response, "content", str(response)))
            return bool(data.get("answerable")) and bool(str(data.get("supporting_quote") or "").strip())
        except Exception as exc:
            logger.warning("Adequacy gate failed closed: %s", exc)
            return False

    def _timed_evidence_is_answerable(
        self,
        question: str,
        evidence: str,
        timings: dict[str, float],
        compare_products: list[str] | None = None,
    ) -> bool:
        started = time.perf_counter()
        try:
            return self._evidence_is_answerable(question, evidence, compare_products)
        finally:
            timings["adequacy"] += time.perf_counter() - started

    @classmethod
    def _compare_evidence_has_entity_coverage(
        cls,
        docs: list[Document],
        sources: list[RetrievedSource],
        compare_products: list[str] | None,
    ) -> bool:
        if not compare_products or len(compare_products) < 2:
            return False
        covered: set[str] = set()
        for doc, source in zip(docs, sources):
            haystack = cls._document_search_text(doc, source)
            body = re.sub(r"\s+", " ", doc.page_content or "").strip()
            if len(body) < 80:
                continue
            for product in compare_products[:2]:
                if cls._entity_in_text(product, haystack):
                    covered.add(cls._normalize_entity_name(product))
        return all(cls._normalize_entity_name(product) in covered for product in compare_products[:2])

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

    @classmethod
    def _compare_entities_supported(cls, compare_products: list[str], supported_entities) -> bool:
        if len(compare_products) < 2:
            return False
        if isinstance(supported_entities, str):
            supported_values = [supported_entities]
        else:
            supported_values = cls._supported_entity_values(supported_entities)
        normalized_supported = [cls._normalize_entity_name(value) for value in supported_values]
        for product in compare_products[:2]:
            normalized_product = cls._normalize_entity_name(product)
            if not any(
                normalized_product == supported
                or normalized_product in supported
                or supported in normalized_product
                for supported in normalized_supported
                if supported
            ):
                return False
        return True

    @staticmethod
    def _supported_entity_values(supported_entities) -> list[str]:
        values: list[str] = []
        for item in list(supported_entities or []):
            if isinstance(item, dict):
                for key in ("entity", "name", "product", "product_name"):
                    if item.get(key):
                        values.append(str(item[key]))
                        break
                else:
                    values.extend(str(value) for value in item.values() if value)
            else:
                values.append(str(item))
        return values

    @staticmethod
    def _normalize_entity_name(value: str) -> str:
        normalized = re.sub(r"(?i)\bautodesk\b", " ", str(value or ""))
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower())
        return re.sub(r"\s+", " ", normalized).strip()

    @classmethod
    def _compare_retrieval_plan(cls, question: str) -> CompareRetrievalPlan:
        if not cls._is_compare_contrast_query(question):
            return CompareRetrievalPlan(False, [], [])
        products = cls._extract_compare_entities(question)
        subqueries = cls._generate_compare_subqueries(question, products)
        return CompareRetrievalPlan(True, products, subqueries)

    @staticmethod
    def _expanded_compare_retrieval_query(question: str, compare_plan: CompareRetrievalPlan) -> str:
        parts = [
            question,
            *compare_plan.subqueries,
        ]
        return "\n".join(part for part in parts if part.strip())

    @staticmethod
    def _is_compare_contrast_query(question: str) -> bool:
        normalized = question.lower()
        patterns = (
            r"\bcompare\b",
            r"\bcontrast\b",
            r"\bdifferences?\s+between\b",
            r"\bvs\.?\b",
            r"\bversus\b",
            r"\bwhich\s+is\s+better\b",
            r"\bwhich\s+should\s+i\s+use\b",
            r"\bshould\s+i\s+use\b",
            r"\bhow\s+does\b.+\bdiffer\s+from\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @classmethod
    def _extract_compare_entities(cls, question: str) -> list[str]:
        compact = re.sub(r"\s+", " ", question).strip(" ?!.")
        capture_patterns = (
            r"\bdifferences?\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$|,| for | when | in )",
            r"\bcompare\s+(.+?)\s+(?:and|with|to)\s+(.+?)(?:\?|$|,| for | when | in )",
            r"\bwhich\s+is\s+better[:,]?\s+(.+?)\s+or\s+(.+?)(?:\?|$|,| for | when | in )",
            r"\bshould\s+i\s+use\s+(.+?)\s+or\s+(.+?)(?:\?|$|,| for | when | in )",
            r"\bwhich\s+should\s+i\s+use[:,]?\s+(.+?)\s+or\s+(.+?)(?:\?|$|,| for | when | in )",
            r"\bhow\s+does\s+(.+?)\s+differ\s+from\s+(.+?)(?:\?|$|,| for | when | in )",
            r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$|,| for | when | in )",
        )
        for pattern in capture_patterns:
            match = re.search(pattern, compact, flags=re.IGNORECASE)
            if match:
                entities = [cls._clean_compare_entity(group) for group in match.groups()]
                return cls._unique_nonempty(entities)[:4]

        candidates = re.findall(
            r"\b(?:[A-Z][A-Za-z0-9]+|[A-Z]{2,}|[0-9]+ds)(?:\s+(?:[A-Z][A-Za-z0-9]+|[A-Z]{2,}|[0-9]+|3D|LT|Max|Cloud))*",
            compact,
        )
        return cls._unique_nonempty(cls._clean_compare_entity(candidate) for candidate in candidates)[:4]

    @staticmethod
    def _clean_compare_entity(value: str) -> str:
        cleaned = re.sub(r"(?i)\b(autodesk|product|software|tool|tools)\b", " ", value)
        cleaned = re.sub(r"(?i)\b(what|which|how|does|do|is|are|the|a|an|use|using|for|when|if|i|we|my|our)\b", " ", cleaned)
        cleaned = re.sub(r"[\"'`“”‘’()\[\]{}:;]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.?/")
        return cleaned

    @classmethod
    def _generate_compare_subqueries(cls, question: str, products: list[str]) -> list[str]:
        if len(products) >= 2:
            left, right = products[0], products[1]
            candidates = [
                f"{left} Autodesk use cases workflows industries target users",
                f"{right} Autodesk use cases workflows industries target users",
                f"{left} {right} Autodesk compare difference interoperability workflow",
                f"{left} {right} Autodesk features 2D 3D modeling documentation collaboration BIM CAD",
            ]
        else:
            candidates = [
                f"{question} Autodesk product comparison use cases workflows",
                f"{question} Autodesk features target users industries",
            ]
        return cls._unique_nonempty(candidates)[:4]

    @staticmethod
    def _unique_nonempty(values) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = re.sub(r"\s+", " ", str(value or "")).strip()
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                unique.append(normalized)
        return unique

    @staticmethod
    def _dedupe_document_pairs(pairs: list[tuple[Document, RetrievedSource]]) -> list[tuple[Document, RetrievedSource]]:
        by_id: dict[str, tuple[Document, RetrievedSource]] = {}
        for doc, source in pairs:
            metadata = doc.metadata or {}
            key = "::".join(
                str(part)
                for part in (
                    metadata.get("chunk_id") or "",
                    metadata.get("source_file") or metadata.get("relative_source_path") or source.source,
                    metadata.get("chunk_index") or "",
                    (doc.page_content or "")[:120],
                )
            )
            current = by_id.get(key)
            if current is None or source.score > current[1].score:
                by_id[key] = (doc, source)
        return list(by_id.values())

    @classmethod
    def _select_balanced_compare_pairs(
        cls,
        pairs: list[tuple[Document, RetrievedSource]],
        products: list[str],
        limit: int,
    ) -> list[tuple[Document, RetrievedSource]]:
        if limit <= 0 or len(products) < 2:
            return pairs[:limit]

        buckets: dict[str, list[tuple[Document, RetrievedSource]]] = {"direct": [], "other": []}
        for product in products[:4]:
            buckets[product] = []

        for pair in pairs:
            doc, source = pair
            haystack = cls._document_search_text(doc, source)
            matched_products = [product for product in products[:4] if cls._entity_in_text(product, haystack)]
            if len(matched_products) >= 2:
                buckets["direct"].append(pair)
            elif len(matched_products) == 1:
                buckets[matched_products[0]].append(pair)
            else:
                buckets["other"].append(pair)

        selected: list[tuple[Document, RetrievedSource]] = []
        selected_ids: set[int] = set()

        # Compare/contrast questions need entity-balanced evidence, otherwise the
        # highest-scoring product page can crowd out the other product before the
        # evidence gate and answer generator ever see it.
        bucket_order = ["direct", *products[:4], "other"]
        while len(selected) < limit:
            added_this_round = False
            for bucket_name in bucket_order:
                bucket = buckets.get(bucket_name, [])
                while bucket and id(bucket[0][0]) in selected_ids:
                    bucket.pop(0)
                if not bucket:
                    continue
                pair = bucket.pop(0)
                selected.append(pair)
                selected_ids.add(id(pair[0]))
                added_this_round = True
                if len(selected) >= limit:
                    break
            if not added_this_round:
                break
        return selected

    @classmethod
    def _ensure_compare_balance_after_rerank(
        cls,
        reranked_docs: list[Document],
        reranked_sources: list[RetrievedSource],
        candidate_docs: list[Document],
        candidate_sources: list[RetrievedSource],
        products: list[str],
        limit: int,
    ) -> tuple[list[Document], list[RetrievedSource]]:
        selected = list(zip(reranked_docs, reranked_sources))[:limit]
        if len(products) < 2 or not selected:
            return [doc for doc, _ in selected], [source for _, source in selected]

        selected_keys = {cls._doc_pair_key(doc, source) for doc, source in selected}
        product_counts = cls._compare_product_counts(selected, products)
        missing_products = [product for product in products[:2] if product_counts.get(product, 0) == 0]
        if not missing_products:
            return [doc for doc, _ in selected], [source for _, source in selected]

        candidates = list(zip(candidate_docs, candidate_sources))
        for missing_product in missing_products:
            replacement = next(
                (
                    (doc, source)
                    for doc, source in candidates
                    if cls._doc_pair_key(doc, source) not in selected_keys
                    and cls._entity_in_text(missing_product, cls._document_search_text(doc, source))
                ),
                None,
            )
            if replacement is None:
                continue
            if len(selected) < limit:
                selected.append(replacement)
            else:
                replace_at = cls._least_needed_compare_index(selected, products, product_counts)
                if replace_at is None:
                    continue
                removed_doc, removed_source = selected[replace_at]
                selected_keys.discard(cls._doc_pair_key(removed_doc, removed_source))
                selected[replace_at] = replacement
            selected_keys.add(cls._doc_pair_key(*replacement))
            product_counts = cls._compare_product_counts(selected, products)

        return [doc for doc, _ in selected], [source for _, source in selected]

    @classmethod
    def _compare_product_counts(cls, pairs: list[tuple[Document, RetrievedSource]], products: list[str]) -> dict[str, int]:
        counts = {product: 0 for product in products[:4]}
        for doc, source in pairs:
            haystack = cls._document_search_text(doc, source)
            for product in products[:4]:
                if cls._entity_in_text(product, haystack):
                    counts[product] += 1
        return counts

    @classmethod
    def _least_needed_compare_index(
        cls,
        selected: list[tuple[Document, RetrievedSource]],
        products: list[str],
        product_counts: dict[str, int],
    ) -> int | None:
        for index in range(len(selected) - 1, -1, -1):
            doc, source = selected[index]
            matched = [product for product in products[:4] if cls._entity_in_text(product, cls._document_search_text(doc, source))]
            if not matched:
                return index
            if all(product_counts.get(product, 0) > 1 for product in matched):
                return index
        return None

    @staticmethod
    def _doc_pair_key(doc: Document, source: RetrievedSource) -> str:
        metadata = doc.metadata or {}
        return "::".join(
            str(part)
            for part in (
                metadata.get("chunk_id") or "",
                metadata.get("source_file") or metadata.get("relative_source_path") or source.source,
                metadata.get("chunk_index") or "",
                (doc.page_content or "")[:120],
            )
        )

    @staticmethod
    def _document_search_text(doc: Document, source: RetrievedSource) -> str:
        metadata = doc.metadata or {}
        metadata_text = " ".join(str(metadata.get(key) or "") for key in ("title", "heading_path", "source_file", "relative_source_path"))
        return f"{metadata_text} {source.source} {source.snippet} {doc.page_content}".lower()

    @staticmethod
    def _entity_in_text(entity: str, text: str) -> bool:
        normalized_entity = re.sub(r"\s+", " ", entity.lower()).strip()
        return bool(normalized_entity) and normalized_entity in text

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
