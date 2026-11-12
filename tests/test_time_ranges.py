from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from agent.time_ranges import resolve_time_range

TZ = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 11, 12, 15, 30, tzinfo=TZ)  # a Thursday
TODAY = datetime(2026, 11, 12, tzinfo=TZ)


def resolve(text):
    found = resolve_time_range(text, NOW)
    assert found is not None, text
    return found.start, found.end


@pytest.mark.parametrize("text, start, end", [
    ("what did I ask yesterday?", TODAY - timedelta(days=1), TODAY),
    ("what did I ask the day before yesterday", TODAY - timedelta(days=2), TODAY - timedelta(days=1)),
    ("what did I ask earlier today", TODAY, NOW),
    ("questions from this morning", TODAY, TODAY + timedelta(hours=12)),
    ("what did we talk about last night", TODAY - timedelta(hours=6), TODAY + timedelta(hours=6)),
    ("anything this week?", datetime(2026, 11, 9, tzinfo=TZ), NOW),
    ("what did I ask last week", datetime(2026, 11, 2, tzinfo=TZ), datetime(2026, 11, 9, tzinfo=TZ)),
    ("last month", datetime(2026, 10, 1, tzinfo=TZ), datetime(2026, 11, 1, tzinfo=TZ)),
    ("in the last 3 days", NOW - timedelta(days=3), NOW),
    ("over the past hour", NOW - timedelta(hours=1), NOW),
    ("2 days ago", TODAY - timedelta(days=2), TODAY - timedelta(days=1)),
    ("a few minutes ago", NOW - timedelta(minutes=6), NOW),
    ("on monday", datetime(2026, 11, 9, tzinfo=TZ), datetime(2026, 11, 10, tzinfo=TZ)),
    ("last thursday", datetime(2026, 11, 5, tzinfo=TZ), datetime(2026, 11, 6, tzinfo=TZ)),
    ("on 2026-11-03", datetime(2026, 11, 3, tzinfo=TZ), datetime(2026, 11, 4, tzinfo=TZ)),
    ("on Nov 3rd", datetime(2026, 11, 3, tzinfo=TZ), datetime(2026, 11, 4, tzinfo=TZ)),
    ("on the 3rd of november", datetime(2026, 11, 3, tzinfo=TZ), datetime(2026, 11, 4, tzinfo=TZ)),
    ("on December 20", datetime(2025, 12, 20, tzinfo=TZ), datetime(2025, 12, 21, tzinfo=TZ)),
])
def test_resolves(text, start, end):
    assert resolve(text) == (start, end)


def test_hours_ago_window_contains_target_and_ends_by_now():
    start, end = resolve("what did I ask 2 hours ago")
    assert start <= NOW - timedelta(hours=2) <= end <= NOW


@pytest.mark.parametrize("text", [
    "what is the capital of France",
    "what was my first question",
    "may I ask something",
    "last time we spoke",
])
def test_no_range(text):
    assert resolve_time_range(text, NOW) is None
