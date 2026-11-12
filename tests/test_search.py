import pytest

from agent.search import CachedSearch, FreeSearch, SearchError, SearchResult


def test_free_search_skips_failing_and_empty_engines():
    def fetch(query, backend, max_results):
        if backend == "a":
            raise RuntimeError("blocked")
        if backend == "b":
            return []
        return [{"title": "T", "href": "https://x", "body": "snippet"}]

    results = FreeSearch(backends=("a", "b", "c"), fetch=fetch).search("q")
    assert results == [SearchResult("T", "https://x", "snippet")]


def test_free_search_returns_fastest_hit_without_waiting_for_slow_engines():
    import threading
    import time

    release = threading.Event()

    def fetch(query, backend, max_results):
        if backend == "slow":
            release.wait(5)
            return [{"title": "slow", "href": "https://slow"}]
        return [{"title": "fast", "href": "https://fast"}]

    started = time.monotonic()
    results = FreeSearch(backends=("slow", "fast"), fetch=fetch).search("q")
    release.set()
    assert results[0].url == "https://fast"
    assert time.monotonic() - started < 2


def test_free_search_raises_when_every_engine_fails():
    def fetch(query, backend, max_results):
        raise RuntimeError(f"{backend} down")

    with pytest.raises(SearchError, match="a down.*b down"):
        FreeSearch(backends=("a", "b"), fetch=fetch).search("q")


class CountingSearch:
    name = "counting"

    def __init__(self):
        self.calls = 0

    def search(self, query):
        self.calls += 1
        return [SearchResult(query, "https://x", "")]


def test_cache_normalises_queries_and_expires():
    now = [0.0]
    inner = CountingSearch()
    cached = CachedSearch(inner, ttl_s=60, clock=lambda: now[0])

    cached.search("Weather  Tokyo")
    cached.search("weather tokyo")
    assert inner.calls == 1

    now[0] = 61
    cached.search("weather tokyo")
    assert inner.calls == 2
