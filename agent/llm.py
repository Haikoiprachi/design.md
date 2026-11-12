"""Gemini client: streaming generation with Google Search grounding, the agent's only tool."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Literal, Protocol


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
    """Emitted once at the end of a response that used Google Search."""
    sources: tuple[Source, ...]
    queries: tuple[str, ...]


LLMEvent = TextChunk | Grounding


class LLM(Protocol):
    def stream(self, system: str, history: list[ChatTurn]) -> Iterator[LLMEvent]: ...


class RateLimitedError(RuntimeError):
    """The Gemini free tier quota was exhausted even after retrying."""


class GeminiLLM:
    _RETRYABLE_CODES = {429, 500, 503, 504}

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        thinking_budget: int | None = None,
        max_retries: int = 3,
    ) -> None:
        from google import genai
        from google.genai import errors, types

        self._types = types
        self._errors = errors
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._thinking_budget = thinking_budget
        self._max_retries = max_retries

    def stream(self, system: str, history: list[ChatTurn]) -> Iterator[LLMEvent]:
        types = self._types
        contents = [types.Content(role=t.role, parts=[types.Part(text=t.text)]) for t in history]
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            thinking_config=(
                types.ThinkingConfig(thinking_budget=self._thinking_budget)
                if self._thinking_budget is not None
                else None
            ),
        )

        for attempt in range(self._max_retries + 1):
            emitted_text = False
            sources: dict[str, Source] = {}
            queries: list[str] = []
            try:
                for chunk in self._client.models.generate_content_stream(
                    model=self._model, contents=contents, config=config
                ):
                    for candidate in chunk.candidates or []:
                        for part in (candidate.content.parts if candidate.content else None) or []:
                            if part.text and not part.thought:
                                emitted_text = True
                                yield TextChunk(part.text)
                        _collect_grounding(candidate.grounding_metadata, sources, queries)
                if sources or queries:
                    yield Grounding(tuple(sources.values()), tuple(queries))
                return
            except self._errors.APIError as exc:
                # Retrying after text was shown would duplicate output, so only retry before that.
                retryable = exc.code in self._RETRYABLE_CODES and not emitted_text
                if not retryable or attempt == self._max_retries:
                    if exc.code == 429:
                        raise RateLimitedError(
                            "Gemini rate limit reached (free tier). Wait a minute and try again."
                        ) from exc
                    raise
                time.sleep(2 ** attempt)


def _collect_grounding(metadata, sources: dict[str, Source], queries: list[str]) -> None:
    if metadata is None:
        return
    for query in metadata.web_search_queries or []:
        if query not in queries:
            queries.append(query)
    for chunk in metadata.grounding_chunks or []:
        web = chunk.web
        if web and web.uri and web.uri not in sources:
            sources[web.uri] = Source(title=web.title or web.uri, uri=web.uri)
