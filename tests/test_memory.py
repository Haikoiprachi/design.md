from datetime import datetime, timedelta, timezone

import pytest

from agent.memory import MemoryStore

T0 = datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def store():
    s = MemoryStore()
    yield s
    s.close()


def test_add_and_recent_are_chronological_per_session(store):
    store.add("a", "user", "one", created_at=T0)
    store.add("b", "user", "other session", created_at=T0)
    store.add("a", "model", "two", created_at=T0 + timedelta(seconds=1))
    store.add("a", "user", "three", created_at=T0 + timedelta(seconds=2))

    assert [m.content for m in store.recent("a", 2)] == ["two", "three"]
    assert [m.content for m in store.recent("a", 10)] == ["one", "two", "three"]


def test_timestamps_round_trip_as_utc(store):
    ist = timezone(timedelta(hours=5, minutes=30))
    saved = store.add("a", "user", "hi", created_at=datetime(2026, 11, 10, 14, 30, 15, 999, tzinfo=ist))
    loaded = store.recent("a", 1)[0]
    assert loaded.created_at == datetime(2026, 11, 10, 9, 0, 15, tzinfo=timezone.utc)
    assert loaded.created_at == saved.created_at


def test_naive_datetimes_are_rejected(store):
    with pytest.raises(ValueError):
        store.add("a", "user", "hi", created_at=datetime(2026, 11, 10))


def test_between_is_half_open(store):
    for hours in range(4):
        store.add("a", "user", f"msg {hours}", created_at=T0 + timedelta(hours=hours))
    rows = store.between(T0 + timedelta(hours=1), T0 + timedelta(hours=3))
    assert [m.content for m in rows] == ["msg 1", "msg 2"]


def test_search_finds_stemmed_words_and_ignores_stopwords(store):
    store.add("a", "user", "How do I bake sourdough bread?", created_at=T0)
    store.add("a", "user", "Weather in Paris", created_at=T0)
    assert [m.content for m in store.search("when did I ask about baking sourdough")] == [
        "How do I bake sourdough bread?"
    ]
    assert store.search("what did I ask you yesterday") == []


def test_search_is_safe_against_fts_syntax(store):
    store.add("a", "user", "quotes test", created_at=T0)
    assert store.search('"quotes" AND (test OR NEAR*') != []


def test_search_excludes_ids(store):
    first = store.add("a", "user", "python packaging", created_at=T0)
    store.add("a", "user", "python typing", created_at=T0)
    assert [m.content for m in store.search("python", exclude_ids={first.id})] == ["python typing"]


def test_stats_and_first_and_latest(store):
    assert store.stats().user_messages == 0
    store.add("a", "user", "first", created_at=T0)
    store.add("a", "model", "answer", created_at=T0)
    store.add("b", "user", "second", created_at=T0 + timedelta(days=1))
    stats = store.stats()
    assert (stats.user_messages, stats.sessions, stats.first_at) == (2, 2, T0)
    assert [m.content for m in store.first_user_messages(1)] == ["first"]
    assert [m.content for m in store.latest_user_messages(5)] == ["first", "second"]


def test_persists_to_disk(tmp_path):
    path = tmp_path / "nested" / "memory.db"
    s = MemoryStore(path)
    s.add("a", "user", "remember me", created_at=T0, sources=[{"title": "x", "uri": "https://x"}])
    s.close()
    reopened = MemoryStore(path)
    message = reopened.recent("a", 1)[0]
    assert (message.content, message.sources) == ("remember me", [{"title": "x", "uri": "https://x"}])
    reopened.close()
