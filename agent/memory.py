"""Persistent, timestamped conversation memory backed by SQLite + FTS5.

Every user and model message is stored with a UTC timestamp. Memory is read by
the orchestrator (not exposed to the model as a tool), so the agent keeps
exactly one tool: web search.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'model')),
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,              -- ISO-8601, UTC, second precision
    sources     TEXT NOT NULL DEFAULT '[]'  -- JSON list of {"title", "uri"}
);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages (created_at);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5 (
    content, content = 'messages', content_rowid = 'id', tokenize = 'porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS messages_after_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts (rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_after_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts (messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
"""

# Words that carry no topical signal for "find earlier messages like this one".
_STOPWORDS = frozenset("""
a about above after again all also am an and any are as ask asked asking at be been before
being but by can could did do does doing done for from get give go had has have having he her
here hers him his how i if in into is it its just know let like me more most my no not now of
on once only or other our out over please question questions remember said say she should so
some tell than that the their them then there these they this those to too up us very was we
were what when where which while who whom why will with would yes you your yours earlier
today yesterday last first previous ago week month time times message messages recall thing
""".split())


@dataclass(frozen=True)
class Message:
    id: int
    session_id: str
    role: str  # "user" | "model"
    content: str
    created_at: datetime  # aware, UTC
    sources: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryStats:
    user_messages: int
    sessions: int
    first_at: datetime | None


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        # One connection shared by the web server's worker threads, so serialise access.
        self._lock = threading.Lock()

    def _fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    def _fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._db.execute(sql, params).fetchone()

    def close(self) -> None:
        self._db.close()

    # ---- writes -------------------------------------------------------------

    def add(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        created_at: datetime | None = None,
        sources: Iterable[dict] = (),
    ) -> Message:
        created = _to_utc(created_at or datetime.now(timezone.utc))
        source_list = list(sources)
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO messages (session_id, role, content, created_at, sources) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, _iso(created), json.dumps(source_list)),
            )
        return Message(cursor.lastrowid, session_id, role, content, created, source_list)

    # ---- reads --------------------------------------------------------------

    def recent(self, session_id: str, limit: int) -> list[Message]:
        """The last `limit` messages of a session, oldest first."""
        rows = self._fetchall(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        return [_row_to_message(r) for r in reversed(rows)]

    def session_messages(self, session_id: str, limit: int = 500) -> list[Message]:
        """A whole conversation, oldest first (used to reopen it in the web UI)."""
        return self.recent(session_id, limit)

    def between(self, start: datetime, end: datetime, limit: int = 40) -> list[Message]:
        """Messages with start <= created_at < end, oldest first."""
        rows = self._fetchall(
            "SELECT * FROM messages WHERE created_at >= ? AND created_at < ? ORDER BY id LIMIT ?",
            (_iso(_to_utc(start)), _iso(_to_utc(end)), limit),
        )
        return [_row_to_message(r) for r in rows]

    def first_user_messages(self, limit: int = 3) -> list[Message]:
        rows = self._fetchall(
            "SELECT * FROM messages WHERE role = 'user' ORDER BY id LIMIT ?", (limit,)
        )
        return [_row_to_message(r) for r in rows]

    def latest_user_messages(self, limit: int, exclude_ids: Iterable[int] = ()) -> list[Message]:
        """Most recent user messages across all sessions, oldest first."""
        excluded = set(exclude_ids)
        rows = self._fetchall(
            "SELECT * FROM messages WHERE role = 'user' ORDER BY id DESC LIMIT ?",
            (limit + len(excluded),),
        )
        kept = [_row_to_message(r) for r in rows if r["id"] not in excluded][:limit]
        return list(reversed(kept))

    def search(self, text: str, limit: int = 5, exclude_ids: Iterable[int] = ()) -> list[Message]:
        """Full-text search over all past messages, best match first."""
        query = _fts_query(text)
        if not query:
            return []
        excluded = set(exclude_ids)
        rows = self._fetchall(
            "SELECT m.* FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid "
            "WHERE messages_fts MATCH ? ORDER BY bm25(messages_fts) LIMIT ?",
            (query, limit + len(excluded)),
        )
        return [_row_to_message(r) for r in rows if r["id"] not in excluded][:limit]

    def stats(self) -> MemoryStats:
        row = self._fetchone(
            "SELECT COUNT(*) AS n, COUNT(DISTINCT session_id) AS s, MIN(created_at) AS first "
            "FROM messages WHERE role = 'user'"
        )
        first = datetime.fromisoformat(row["first"]) if row["first"] else None
        return MemoryStats(user_messages=row["n"], sessions=row["s"], first_at=first)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("datetimes stored in memory must be timezone-aware")
    return dt.astimezone(timezone.utc).replace(microsecond=0)


def _iso(dt: datetime) -> str:
    # One fixed format, so string comparison in SQL matches chronological order.
    return dt.isoformat(timespec="seconds")


def _row_to_message(row: sqlite3.Row) -> Message:
    return Message(
        id=row["id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        created_at=datetime.fromisoformat(row["created_at"]),
        sources=json.loads(row["sources"]),
    )


def _fts_query(text: str) -> str:
    words = {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOPWORDS}
    # Quote each term so user text can't inject FTS syntax; OR keeps recall high, bm25 ranks.
    return " OR ".join(f'"{w}"' for w in sorted(words))
