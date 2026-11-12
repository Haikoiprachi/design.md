"""The agent loop: build context from memory, stream the model's answer, remember the turn."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import Callable, Iterator

from .llm import LLM, ChatTurn, Grounding, Source, TextChunk
from .memory import MemoryStore, Message
from .prompts import build_system_prompt, format_timestamp, snippet, strip_timestamp_prefix
from .time_ranges import resolve_time_range

# Questions about the conversation itself ("what did I ask yesterday?").
_MEMORY_INTENT = re.compile(
    r"\b("
    r"did i|have i|had i|i (?:asked|said|told|mentioned|wrote|searched)|"
    r"we (?:talked|discussed|spoke|chatted)|"
    r"(?:my|the) (?:first|last|previous|earlier|prior|latest|recent) (?:questions?|messages?|quer(?:y|ies)|requests?)|"
    r"remember|recall|last time|earlier|so far|history|"
    r"how many (?:questions|messages|times)"
    r")\b",
    re.IGNORECASE,
)
_FIRST_INTENT = re.compile(r"\b(first|earliest)\b", re.IGNORECASE)

# Long enough to recognise a "[Thu 2026-11-12 14:03]" prefix before streaming it.
_PREFIX_BUFFER_CHARS = 24


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class TurnResult:
    answer: str
    sources: tuple[Source, ...]
    search_queries: tuple[str, ...]
    ttft_s: float | None  # time to first visible token
    total_s: float

    @property
    def searched(self) -> bool:
        return bool(self.search_queries or self.sources)


AgentEvent = TextDelta | TurnResult


class ChatAgent:
    def __init__(
        self,
        memory: MemoryStore,
        llm: LLM,
        *,
        tz: tzinfo,
        session_id: str | None = None,
        history_turns: int = 12,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.memory = memory
        self.llm = llm
        self.tz = tz
        self.session_id = session_id or _new_session_id()
        self.history_turns = history_turns
        self._clock = clock or (lambda: datetime.now(tz))
        self.last_system_prompt = ""

    def new_session(self) -> None:
        self.session_id = _new_session_id()

    def chat(self, user_text: str) -> Iterator[AgentEvent]:
        now = self._clock()
        system, history = self.build_context(user_text, now)
        self.last_system_prompt = system
        self.memory.add(self.session_id, "user", user_text, created_at=now)

        started = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []
        pending = ""  # held back until we know it isn't a copied timestamp prefix
        grounding = Grounding((), ())

        for event in self.llm.stream(system, history):
            if isinstance(event, Grounding):
                grounding = event
                continue
            if not parts and len(pending) < _PREFIX_BUFFER_CHARS:
                pending += event.text
                if len(pending) < _PREFIX_BUFFER_CHARS:
                    continue
                text, pending = strip_timestamp_prefix(pending), ""
            else:
                text = event.text
            if text:
                ttft = ttft if ttft is not None else time.perf_counter() - started
                parts.append(text)
                yield TextDelta(text)

        if pending and (text := strip_timestamp_prefix(pending)):
            ttft = ttft if ttft is not None else time.perf_counter() - started
            parts.append(text)
            yield TextDelta(text)

        answer = "".join(parts).strip()
        self.memory.add(
            self.session_id,
            "model",
            answer or "(no answer)",
            sources=[{"title": s.title, "uri": s.uri} for s in grounding.sources],
        )
        yield TurnResult(
            answer=answer,
            sources=grounding.sources,
            search_queries=grounding.queries,
            ttft_s=ttft,
            total_s=time.perf_counter() - started,
        )

    # ---- context ------------------------------------------------------------

    def build_context(self, user_text: str, now: datetime) -> tuple[str, list[ChatTurn]]:
        recent = self.memory.recent(self.session_id, self.history_turns)
        history = [ChatTurn(m.role, self._render_turn(m)) for m in recent]
        history.append(ChatTurn("user", f"[{format_timestamp(now, self.tz)}] {user_text}"))
        memory_block = self._memory_block(user_text, now, {m.id for m in recent})
        return build_system_prompt(now, memory_block), history

    def _render_turn(self, message: Message) -> str:
        if message.role == "user":
            return f"[{format_timestamp(message.created_at, self.tz)}] {message.content}"
        return message.content

    def _memory_block(self, user_text: str, now: datetime, in_history: set[int]) -> str:
        stats = self.memory.stats()
        if stats.user_messages == 0:
            return "MEMORY\nThis is your first conversation with the user; there are no earlier messages."

        sections = [
            "MEMORY",
            f"On record: {stats.user_messages} user messages across {stats.sessions} session(s), "
            f"the first on {format_timestamp(stats.first_at, self.tz)}.",
        ]
        shown = set(in_history)

        if _MEMORY_INTENT.search(user_text):
            if time_range := resolve_time_range(user_text, now):
                rows = self.memory.between(time_range.start, time_range.end)
                header = (
                    f'Every message from "{time_range.label}" '
                    f"({format_timestamp(time_range.start, self.tz)} to "
                    f"{format_timestamp(time_range.end, self.tz)}):"
                )
                sections.append(self._render_section(header, rows, empty="(none: no messages in this period)"))
                shown.update(m.id for m in rows)
            if _FIRST_INTENT.search(user_text):
                rows = self.memory.first_user_messages(3)
                sections.append(self._render_section("The user's first messages ever:", rows))
                shown.update(m.id for m in rows)
            rows = self.memory.latest_user_messages(5, exclude_ids=shown)
            sections.append(self._render_section("Most recent earlier user messages:", rows))
            shown.update(m.id for m in rows)

        related = self.memory.search(user_text, limit=5, exclude_ids=shown)
        sections.append(self._render_section("Possibly related earlier messages:", related))
        return "\n\n".join(s for s in sections if s)

    def _render_section(self, header: str, rows: list[Message], empty: str = "") -> str:
        if not rows:
            return f"{header}\n{empty}" if empty else ""
        lines = [
            f"- [{format_timestamp(m.created_at, self.tz)}] "
            f"{'user' if m.role == 'user' else 'you'}: {snippet(m.content, 300 if m.role == 'user' else 200)}"
            for m in rows
        ]
        return "\n".join([header, *lines])


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]
