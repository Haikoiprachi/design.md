import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from agent.llm import Grounding, RateLimitedError, SearchStarted, Source, TextChunk
from agent.memory import MemoryStore
from web import create_app

TZ = ZoneInfo("Asia/Kolkata")


class ScriptedLLM:
    model = "fake-model"

    def __init__(self, events=None, error=None):
        self.events = events or [TextChunk("Hello there, this is a reply.")]
        self.error = error
        self.histories = []

    def stream(self, system, history):
        self.histories.append(history)
        if self.error:
            raise self.error
        yield from self.events


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def memory():
    store = MemoryStore()
    yield store
    store.close()


def client_for(memory, llm):
    return TestClient(create_app(memory, llm, tz=TZ))


def test_index_page_is_served(memory):
    response = client_for(memory, ScriptedLLM()).get("/")
    assert response.status_code == 200
    assert "Gemini Chat Agent" in response.text


def test_chat_streams_session_search_text_and_done(memory):
    llm = ScriptedLLM([
        SearchStarted(("ceo microsoft",)),
        TextChunk("Satya Nadella is the CEO of Microsoft [1]."),
        Grounding((Source("Wiki", "https://w"),), ("ceo microsoft",)),
    ])
    response = client_for(memory, llm).post("/api/chat", json={"message": "Who runs Microsoft?"})

    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names == ["session", "search", "delta", "done"]
    assert events[1][1] == {"queries": ["ceo microsoft"]}
    done = events[-1][1]
    assert done["answer"] == "Satya Nadella is the CEO of Microsoft [1]."
    assert done["sources"] == [{"title": "Wiki", "uri": "https://w"}]
    assert done["queries"] == ["ceo microsoft"]
    assert done["model"] == "fake-model"


def test_follow_up_with_session_id_continues_the_conversation(memory):
    llm = ScriptedLLM()
    client = client_for(memory, llm)
    first = parse_sse(client.post("/api/chat", json={"message": "first"}).text)
    session_id = first[0][1]["session_id"]

    client.post("/api/chat", json={"message": "second", "session_id": session_id})

    second_history = llm.histories[-1]
    assert [t.role for t in second_history] == ["user", "model", "user"]
    assert second_history[0].text.endswith("first")


def test_history_lists_questions_newest_first_and_session_reopens(memory):
    old = datetime.now(timezone.utc) - timedelta(days=1)
    memory.add("abc123", "user", "older question", created_at=old)
    memory.add("abc123", "model", "older answer", created_at=old, sources=[{"title": "T", "uri": "https://t"}])
    memory.add("def456", "user", "newer question")
    client = client_for(memory, ScriptedLLM())

    questions = client.get("/api/history").json()["questions"]
    assert [q["content"] for q in questions] == ["newer question", "older question"]
    assert questions[1]["session_id"] == "abc123"
    assert questions[1]["created_at"].endswith("+00:00")

    session = client.get("/api/sessions/abc123").json()
    assert [(m["role"], m["content"]) for m in session["messages"]] == [
        ("user", "older question"), ("model", "older answer")
    ]
    assert session["messages"][1]["sources"] == [{"title": "T", "uri": "https://t"}]
    assert client.get("/api/sessions/ffffff").status_code == 404


@pytest.mark.parametrize("body", [
    {"message": ""},
    {"message": "x" * 4001},
    {"message": "hi", "session_id": "../../etc"},
])
def test_invalid_requests_are_rejected(memory, body):
    assert client_for(memory, ScriptedLLM()).post("/api/chat", json=body).status_code == 422


def test_rate_limit_becomes_an_error_event_not_a_crash(memory):
    llm = ScriptedLLM(error=RateLimitedError("Gemini rate limit reached"))
    events = parse_sse(client_for(memory, llm).post("/api/chat", json={"message": "hi"}).text)
    assert events[-1] == ("error", {"message": "Gemini rate limit reached"})
