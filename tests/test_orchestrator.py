from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from agent.llm import Grounding, Source, TextChunk
from agent.memory import MemoryStore
from agent.orchestrator import ChatAgent, TextDelta, TurnResult

TZ = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 11, 12, 15, 30, tzinfo=TZ)


class FakeLLM:
    """Records what it was sent and replies with scripted chunks."""

    def __init__(self, chunks=("Hello", " there"), grounding=None):
        self.chunks = chunks
        self.grounding = grounding
        self.calls = []

    def stream(self, system, history):
        self.calls.append((system, history))
        for chunk in self.chunks:
            yield TextChunk(chunk)
        if self.grounding:
            yield self.grounding


@pytest.fixture
def memory():
    store = MemoryStore()
    yield store
    store.close()


def make_agent(memory, llm, clock=lambda: NOW):
    return ChatAgent(memory, llm, tz=TZ, session_id="current", clock=clock)


def run(agent, text):
    events = list(agent.chat(text))
    deltas = "".join(e.text for e in events if isinstance(e, TextDelta))
    result = events[-1]
    assert isinstance(result, TurnResult)
    return deltas, result


def test_streams_answer_and_stores_both_turns_with_sources(memory):
    grounding = Grounding((Source("example.com", "https://example.com"),), ("weather tokyo",))
    agent = make_agent(memory, FakeLLM(("It is ", "sunny."), grounding))

    deltas, result = run(agent, "Weather in Tokyo?")

    assert deltas == result.answer == "It is sunny."
    assert result.searched and result.search_queries == ("weather tokyo",)
    stored = memory.recent("current", 10)
    assert [(m.role, m.content) for m in stored] == [("user", "Weather in Tokyo?"), ("model", "It is sunny.")]
    assert stored[0].created_at == NOW
    assert stored[1].sources == [{"title": "example.com", "uri": "https://example.com"}]


def test_prompt_contains_current_time_and_history_is_timestamped(memory):
    llm = FakeLLM()
    agent = make_agent(memory, llm)
    run(agent, "first")
    run(agent, "second")

    system, history = llm.calls[-1]
    assert "Thursday, November 12, 2026, 15:30" in system
    assert "Asia/Kolkata" in system
    assert [t.role for t in history] == ["user", "model", "user"]
    assert history[0].text == "[Thu 2026-11-12 15:30] first"
    assert history[-1].text == "[Thu 2026-11-12 15:30] second"


def test_time_question_injects_exact_rows_from_that_period(memory):
    yesterday = NOW - timedelta(days=1)
    memory.add("old", "user", "Population of Iceland?", created_at=yesterday)
    memory.add("old", "user", "Something from last week", created_at=NOW - timedelta(days=8))
    llm = FakeLLM()

    run(make_agent(memory, llm), "What did I ask you yesterday?")

    system = llm.calls[0][0]
    period = system.split('Every message from "yesterday"')[1].split("\n\n")[0]
    assert "[Wed 2026-11-11 15:30] user: Population of Iceland?" in period
    assert "last week" not in period


def test_empty_period_is_stated_explicitly(memory):
    memory.add("old", "user", "pasta recipe", created_at=NOW - timedelta(minutes=30))
    llm = FakeLLM()
    run(make_agent(memory, llm), "Did I ask about stocks last week?")
    assert "(none: no messages in this period)" in llm.calls[0][0]


def test_first_question_and_related_memory_across_sessions(memory):
    memory.add("s1", "user", "My name is Riya", created_at=NOW - timedelta(days=5))
    memory.add("s2", "user", "Tallest mountain?", created_at=NOW - timedelta(days=1))
    llm = FakeLLM()
    agent = make_agent(memory, llm)

    run(agent, "What was my first question?")
    assert "The user's first messages ever:\n- [Sat 2026-11-07 15:30] user: My name is Riya" in llm.calls[0][0]

    run(agent, "What's my name?")
    assert "Possibly related earlier messages:\n- [Sat 2026-11-07 15:30] user: My name is Riya" in llm.calls[1][0]


@pytest.mark.parametrize("question", [
    "What was the very first question I ever asked you?",
    "What was my first question?",
    "what's the first thing I asked you",
])
def test_first_question_phrasings_pull_first_messages(memory, question):
    memory.add("s1", "user", "Recommend a sci-fi novel about Mars", created_at=NOW - timedelta(days=9))
    llm = FakeLLM()
    run(make_agent(memory, llm), question)
    assert "The user's first messages ever:" in llm.calls[0][0]


def test_first_conversation_says_so(memory):
    llm = FakeLLM()
    run(make_agent(memory, llm), "hi")
    assert "first conversation" in llm.calls[0][0]


def test_copied_timestamp_prefix_is_stripped_even_across_chunks(memory):
    agent = make_agent(memory, FakeLLM(("[Thu 2026-11", "-12 15:30] You asked", " about Iceland.")))
    deltas, result = run(agent, "what did I ask?")
    assert deltas == result.answer == "You asked about Iceland."


def test_short_answers_are_not_lost_by_prefix_buffering(memory):
    deltas, result = run(make_agent(memory, FakeLLM(("391",))), "17*23?")
    assert deltas == result.answer == "391"
    assert result.ttft_s is not None


def test_new_session_keeps_long_term_memory(memory):
    llm = FakeLLM()
    agent = make_agent(memory, llm)
    run(agent, "I love hiking in Colorado")
    agent.new_session()
    run(agent, "Any hiking tips?")

    system, history = llm.calls[-1]
    assert len(history) == 1  # fresh session: no short-term history
    assert "I love hiking in Colorado" in system  # but long-term memory still finds it
