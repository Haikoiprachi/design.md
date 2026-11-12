"""The agent's one tool: web search.

Two free providers behind one interface:
- TavilySearch: a search API built for LLM agents (free tier, needs TAVILY_API_KEY). Fast and reliable.
- FreeSearch: no key at all; tries several public engines via the `ddgs` package until one answers.
Both are wrapped in a short-lived cache so repeated or follow-up searches cost nothing.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchError(RuntimeError):
    """No provider could return results."""


class WebSearch(Protocol):
    name: str

    def search(self, query: str) -> list[SearchResult]: ...


class TavilySearch:
    name = "tavily"
    _URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, *, max_results: int = 5, timeout_s: float = 10.0) -> None:
        import httpx

        self._client = httpx.Client(timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"})
        self._max_results = max_results

    def search(self, query: str) -> list[SearchResult]:
        try:
            response = self._client.post(
                self._URL, json={"query": query, "max_results": self._max_results, "search_depth": "basic"}
            )
            response.raise_for_status()
        except Exception as exc:
            raise SearchError(f"Tavily search failed: {exc}") from exc
        return [
            SearchResult(r.get("title") or r["url"], r["url"], r.get("content", ""))
            for r in response.json().get("results", [])
        ]


class FreeSearch:
    """Keyless search. Public engines rate-limit scrapers unpredictably, so race several of them."""

    name = "free (ddgs)"
    DEFAULT_BACKENDS = ("yahoo", "duckduckgo", "bing")

    def __init__(
        self,
        *,
        backends: tuple[str, ...] = DEFAULT_BACKENDS,
        max_results: int = 5,
        timeout_s: int = 6,
        fetch: Callable[[str, str, int], list[dict]] | None = None,
    ) -> None:
        self._backends = backends
        self._max_results = max_results
        self._timeout_s = timeout_s
        self._fetch = fetch or self._ddgs_fetch

    def search(self, query: str) -> list[SearchResult]:
        """Ask every engine at once and return the first non-empty answer (latency = fastest hit)."""
        errors = []
        pool = ThreadPoolExecutor(max_workers=len(self._backends))
        futures = {pool.submit(self._fetch, query, b, self._max_results): b for b in self._backends}
        try:
            for future in as_completed(futures):
                backend = futures[future]
                try:
                    rows = future.result()
                except Exception as exc:  # each engine fails in its own way; just move on
                    errors.append(f"{backend}: {exc}")
                    continue
                results = [SearchResult(r.get("title") or r["href"], r["href"], r.get("body", "")) for r in rows]
                if results:
                    return results
                errors.append(f"{backend}: no results")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)  # don't wait for slower engines
        raise SearchError("No search engine returned results (" + "; ".join(sorted(errors)) + ")")

    def _ddgs_fetch(self, query: str, backend: str, max_results: int) -> list[dict]:
        from ddgs import DDGS

        return DDGS(timeout=self._timeout_s).text(query, max_results=max_results, backend=backend)


class CachedSearch:
    """Remembers results for a few minutes, keyed by the normalised query."""

    def __init__(self, inner: WebSearch, *, ttl_s: float = 600, clock: Callable[[], float] = time.monotonic):
        self._inner = inner
        self._ttl_s = ttl_s
        self._clock = clock
        self._cache: dict[str, tuple[float, list[SearchResult]]] = {}
        self.name = inner.name

    def search(self, query: str) -> list[SearchResult]:
        key = " ".join(query.lower().split())
        hit = self._cache.get(key)
        if hit and self._clock() - hit[0] < self._ttl_s:
            return hit[1]
        results = self._inner.search(query)
        self._cache[key] = (self._clock(), results)
        return results


def build_search(tavily_api_key: str | None) -> WebSearch:
    provider = TavilySearch(tavily_api_key) if tavily_api_key else FreeSearch()
    return CachedSearch(provider)
