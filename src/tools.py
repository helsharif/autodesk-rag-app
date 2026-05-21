"""External tools used by the RAG agent."""

from __future__ import annotations

import requests

from src.config import get_settings


def web_search(query: str) -> str:
    settings = get_settings()
    if not settings.serpapi_api_key:
        return "No web search results found. SERPAPI_API_KEY is not configured."
    response = requests.get(
        "https://serpapi.com/search.json",
        params={"engine": "google", "q": query, "api_key": settings.serpapi_api_key, "num": 5, "safe": "active"},
        timeout=20,
    )
    response.raise_for_status()
    results = response.json().get("organic_results", [])
    if not results:
        return "No web search results found."
    blocks = []
    for index, item in enumerate(results, start=1):
        blocks.append(f"[Web {index}] {item.get('title', 'Untitled')}\n{item.get('link', '')}\n{item.get('snippet', '')}")
    return "\n\n".join(blocks)
