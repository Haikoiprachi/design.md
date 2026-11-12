"""Web chat UI: a small FastAPI server that streams the agent's answers over Server-Sent Events.

Run:  python web.py      then open http://localhost:8000
The page is static/index.html (plain HTML/CSS/JS, no build step). The agent, memory and
search are the same objects the terminal app uses.
"""

from __future__ import annotations

import json
import os
from datetime import tzinfo
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.llm import LLM, RateLimitedError, SearchStarted
from agent.memory import MemoryStore, Message
from agent.orchestrator import ChatAgent, TextDelta, TurnResult

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_ID_PATTERN = r"^[a-f0-9]{6,32}$"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)


def create_app(memory: MemoryStore, llm: LLM, *, tz: tzinfo, history_turns: int = 12) -> FastAPI:
    app = FastAPI(title="Gemini Chat Agent", docs_url=None, redoc_url=None)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> StreamingResponse:
        agent = ChatAgent(memory, llm, tz=tz, session_id=request.session_id, history_turns=history_turns)
        return StreamingResponse(
            _stream_turn(agent, request.message.strip(), getattr(llm, "model", "")),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/history")
    def history(limit: int = Query(default=100, ge=1, le=500)) -> dict:
        """The user's past questions, newest first, for the sidebar."""
        questions = memory.latest_user_messages(limit)
        return {"questions": [_message_json(m) for m in reversed(questions)]}

    @app.get("/api/sessions/{session_id}")
    def session(session_id: str) -> dict:
        messages = memory.session_messages(session_id)
        if not messages:
            raise HTTPException(status_code=404, detail="No such conversation")
        return {"session_id": session_id, "messages": [_message_json(m) for m in messages]}

    return app


def _stream_turn(agent: ChatAgent, message: str, model: str) -> Iterator[str]:
    yield _sse("session", {"session_id": agent.session_id})
    try:
        for event in agent.chat(message):
            if isinstance(event, TextDelta):
                yield _sse("delta", {"text": event.text})
            elif isinstance(event, SearchStarted):
                yield _sse("search", {"queries": list(event.queries)})
            elif isinstance(event, TurnResult):
                yield _sse("done", {
                    "answer": event.answer,
                    "sources": [{"title": s.title, "uri": s.uri} for s in event.sources],
                    "queries": list(event.search_queries),
                    "ttft_s": event.ttft_s,
                    "total_s": event.total_s,
                    "model": getattr(agent.llm, "model", model),
                })
    except RateLimitedError as exc:
        yield _sse("error", {"message": str(exc)})
    except Exception as exc:  # report to the page instead of dropping the connection
        yield _sse("error", {"message": f"Something went wrong: {exc}"})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _message_json(message: Message) -> dict:
    return {
        "id": message.id,
        "session_id": message.session_id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at.isoformat(),  # UTC; the browser shows local time
        "sources": message.sources,
    }


def main() -> None:
    import uvicorn

    from agent.config import load_settings
    from agent.llm import GeminiLLM
    from agent.search import build_search

    settings = load_settings()
    search = build_search(settings.tavily_api_key)
    llm = GeminiLLM(
        settings.api_key, settings.model, search,
        fallback_models=settings.fallback_models, thinking_budget=settings.thinking_budget,
    )
    app = create_app(MemoryStore(settings.db_path), llm, tz=settings.timezone,
                     history_turns=settings.history_turns)
    port = int(os.getenv("PORT", "8000"))
    print(f"Chat UI running at http://localhost:{port}  (model {settings.model}, search: {search.name})")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
