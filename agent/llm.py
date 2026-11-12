"""Gemini client: streaming generation with one function tool, `web_search`.

Gemini decides when to search. When it calls web_search we run the query (several in
parallel if it asks for several), number the results, send them back, and keep
streaming. Result numbers are global for the turn, so the model's [n] citations match
the sources list shown to the user.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Protocol

from .search import SearchError, WebSearch


@dataclass(frozen=True)
class ChatTurn:
    role: Literal["user", "model"]
    text: str


@dataclass(frozen=True)
class TextChunk:
    text: str


@dataclass(frozen=True)
class Source:
    title: str
    uri: str


@dataclass(frozen=True)
class Grounding:
    """Emitted once at the end of a turn that searched. sources[i] is citation [i + 1]."""
    sources: tuple[Source, ...]
    queries: tuple[str, ...]


LLMEvent = TextChunk | Grounding


def _transient_network_errors() -> tuple[type[BaseException], ...]:
    """Timeouts/connection errors from httpx and httpx2 (google-genai may use either)."""
    import importlib

    found = []
    for module_name in ("httpx", "httpx2"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        found += [getattr(module, name) for name in ("TimeoutException", "ConnectError") if hasattr(module, name)]
    return tuple(found)


class LLM(Protocol):
    def stream(self, system: str, history: list[ChatTurn]) -> Iterator[LLMEvent]: ...


class RateLimitedError(RuntimeError):
    """The Gemini free tier quota was exhausted even after retrying."""


WEB_SEARCH_DECLARATION = {
    "name": "web_search",
    "description": (
        "Search the web for current or uncertain information. Returns numbered results "
        "(id, title, url, snippet). Cite them in your answer as [id]."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "A specific, standalone search query (resolve pronouns from the conversation).",
            }
        },
        "required": ["query"],
    },
}


class GeminiLLM:
    _RETRYABLE_CODES = {429, 500, 503, 504}

    def __init__(
        self,
        api_key: str,
        model: str,
        search: WebSearch,
        *,
        fallback_models: tuple[str, ...] = (),
        thinking_budget: int | None = None,
        max_retries: int = 3,
        max_search_rounds: int = 3,
        request_timeout_s: float = 30,
        client: Any = None,
    ) -> None:
        from google import genai
        from google.genai import errors, types

        self._types = types
        self._errors = errors
        self._network_errors = _transient_network_errors()
        # The SDK's own retries are turned off: they wait silently on an overloaded model,
        # while we would rather fail over to the next model immediately.
        self._client = client or genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1), timeout=int(request_timeout_s * 1000)
            ),
        )
        self._models = (model, *(m for m in fallback_models if m != model))
        self._search = search
        self._thinking_budget = thinking_budget
        self._max_retries = max_retries
        self._max_search_rounds = max_search_rounds

    def stream(self, system: str, history: list[ChatTurn]) -> Iterator[LLMEvent]:
        types = self._types
        contents = [types.Content(role=t.role, parts=[types.Part(text=t.text)]) for t in history]
        sources: list[Source] = []
        queries: list[str] = []

        for round_index in range(self._max_search_rounds + 1):
            # On the last round, tools stay declared (history contains calls) but calling is disabled.
            allow_search = round_index < self._max_search_rounds
            calls, model_parts = yield from self._stream_round(contents, self._config(system, allow_search))
            if not calls and not _has_text(model_parts):
                # Models occasionally end a turn with nothing to say; ask once more for a direct answer.
                calls, model_parts = yield from self._stream_round(contents, self._config(system, False))
            if not calls:
                break
            # Send the model's own parts back unchanged: they carry the thought signatures Gemini needs.
            contents.append(types.Content(role="model", parts=model_parts))
            contents.append(types.Content(role="user", parts=self._run_searches(calls, sources, queries)))

        if sources or queries:
            yield Grounding(tuple(sources), tuple(queries))

    def _config(self, system: str, allow_search: bool):
        types = self._types
        return types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=[WEB_SEARCH_DECLARATION])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO" if allow_search else "NONE")
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=(
                types.ThinkingConfig(thinking_budget=self._thinking_budget)
                if self._thinking_budget is not None
                else None
            ),
        )

    @property
    def model(self) -> str:
        return self._models[0]

    def _stream_round(self, contents, config):
        """Stream one model response. Yields text; returns (function_calls, all_parts).

        On an overload/timeout before any text, fail over to the next model; back off
        only after every model has been tried. A model that answers moves to the front.
        """
        attempts = len(self._models) * (self._max_retries + 1)
        for attempt in range(attempts):
            model = self._models[0]
            emitted_text = False
            calls, parts = [], []
            try:
                for chunk in self._client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                ):
                    for candidate in chunk.candidates or []:
                        for part in (candidate.content.parts if candidate.content else None) or []:
                            parts.append(part)
                            if part.function_call:
                                calls.append(part.function_call)
                            elif part.text and not part.thought:
                                emitted_text = True
                                yield TextChunk(part.text)
                return calls, parts
            except (self._errors.APIError, *self._network_errors) as exc:
                code = getattr(exc, "code", 504)  # timeouts/connection drops behave like a 504
                # Retrying after text was shown would duplicate output, so only retry before that.
                retryable = code in self._RETRYABLE_CODES and not emitted_text
                if not retryable or attempt == attempts - 1:
                    if code == 429:
                        raise RateLimitedError(
                            "Gemini rate limit reached (free tier). Wait a minute and try again."
                        ) from exc
                    raise
                self._models = (*self._models[1:], model)  # rotate: try the next model now
                if (attempt + 1) % len(self._models) == 0:
                    time.sleep(2 ** ((attempt + 1) // len(self._models) - 1))
        raise AssertionError("unreachable")

    def _run_searches(self, calls, sources: list[Source], queries: list[str]) -> list:
        types = self._types
        call_queries = [str((call.args or {}).get("query", "")).strip() for call in calls]
        with ThreadPoolExecutor(max_workers=max(1, len(calls))) as pool:
            outcomes = list(pool.map(self._safe_search, call_queries))

        responses = []
        for call, query, outcome in zip(calls, call_queries, outcomes):
            if call.name != "web_search":
                payload = {"error": f"Unknown tool {call.name!r}. The only tool is web_search."}
            elif isinstance(outcome, str):
                payload = {"error": outcome}
            else:
                queries.append(query)
                payload = {"results": [
                    {"id": _source_id(sources, r.title, r.url), "title": r.title, "url": r.url, "snippet": r.snippet}
                    for r in outcome
                ]}
            responses.append(types.Part(function_response=types.FunctionResponse(
                id=call.id, name=call.name, response=payload
            )))
        return responses

    def _safe_search(self, query: str):
        if not query:
            return "Empty query."
        try:
            return self._search.search(query)
        except SearchError as exc:
            return f"Search failed: {exc}. Answer from your own knowledge and say it is unverified."


def _has_text(parts) -> bool:
    return any(part.text and not part.thought for part in parts)


def _source_id(sources: list[Source], title: str, url: str) -> int:
    """1-based citation number for url, adding it to `sources` if new."""
    for index, source in enumerate(sources, start=1):
        if source.uri == url:
            return index
    sources.append(Source(title=title, uri=url))
    return len(sources)
