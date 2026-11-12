"""Turn natural-language time references ("yesterday", "3 hours ago") into concrete ranges.

The orchestrator uses this to answer "what did I ask you <when>?" with an exact
database query, instead of hoping the model can pick the right rows out of a
long history on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable


@dataclass(frozen=True)
class TimeRange:
    start: datetime
    end: datetime
    label: str


_NUMBER = (
    r"(\d+|a couple of|couple of|a few|few|an|a|one|two|three|four|five|six|seven|eight|nine|ten)"
)
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "few": 3, "a few": 3, "couple of": 2, "a couple of": 2,
}
_UNIT = r"(minute|min|hour|hr|day|week|month)s?"
_UNIT_ALIASES = {"min": "minute", "hr": "hour"}
_UNITS = {
    "minute": timedelta(minutes=1),
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
    "month": timedelta(days=30),
}
_MONTH = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1
)}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def resolve_time_range(text: str, now: datetime) -> TimeRange | None:
    """Return the first time range referenced in `text`, relative to the aware datetime `now`."""
    lowered = text.lower()
    for resolver in _RESOLVERS:
        found = resolver(lowered, now)
        if found:
            return found
    return None


def _day_start(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _whole_day(start: datetime, label: str) -> TimeRange:
    return TimeRange(start, start + timedelta(days=1), label)


def _parse_number(token: str) -> int:
    return int(token) if token.isdigit() else _NUMBER_WORDS[token]


def _parse_unit(token: str) -> str:
    return _UNIT_ALIASES.get(token, token)


def _units_ago(text: str, now: datetime) -> TimeRange | None:
    """'3 hours ago', 'a few minutes ago', '2 days ago'."""
    match = re.search(rf"\b{_NUMBER}\s+{_UNIT}\s+ago\b", text)
    if not match:
        return None
    count = _parse_number(match.group(1))
    unit_name = _parse_unit(match.group(2))
    label = match.group(0)
    if unit_name == "day":
        return _whole_day(_day_start(now) - timedelta(days=count), label)

    unit = _UNITS[unit_name]
    if "few" in match.group(1) or "couple" in match.group(1):
        return TimeRange(now - unit * (2 * count), now, label)
    # People are loose with "an hour ago", so the window is deliberately generous;
    # the model still sees exact timestamps and can filter further.
    center = now - unit * count
    half_width = unit * max(0.5 * count, 1)
    return TimeRange(center - half_width, min(center + half_width, now), label)


def _rolling_window(text: str, now: datetime) -> TimeRange | None:
    """'in the last 3 days', 'past hour', 'the last week' (rolling, ending now)."""
    match = re.search(rf"\b(?:the\s+)?(?:last|past)\s+{_NUMBER}\s+{_UNIT}\b", text)
    if match:
        count = _parse_number(match.group(1))
        unit = _UNITS[_parse_unit(match.group(2))]
        return TimeRange(now - unit * count, now, match.group(0))
    # Without a number we need "the last"/"past", so plain "last week" stays a calendar week.
    match = re.search(rf"\b(?:the\s+(?:last|past)|past)\s+{_UNIT}\b", text)
    if match:
        unit = _UNITS[_parse_unit(match.group(1))]
        return TimeRange(now - unit, now, match.group(0))
    return None


def _explicit_date(text: str, now: datetime) -> TimeRange | None:
    """'2026-11-10', 'Nov 10', '10th of November'."""
    year, month, day = None, None, None
    if match := re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text):
        year, month, day = (int(g) for g in match.groups())
    elif match := re.search(rf"\b{_MONTH}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", text):
        month, day = _MONTHS[match.group(1)[:3]], int(match.group(2))
    elif match := re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}\b", text):
        day, month = int(match.group(1)), _MONTHS[match.group(2)[:3]]
    else:
        return None

    try:
        start = now.replace(year=year or now.year, month=month, day=day,
                            hour=0, minute=0, second=0, microsecond=0)
    except ValueError:
        return None
    if year is None and start > now:  # "Dec 20" asked in January means last year
        start = start.replace(year=start.year - 1)
    return _whole_day(start, match.group(0))


def _weekday(text: str, now: datetime) -> TimeRange | None:
    """'on monday', 'last friday' -> the most recent such day before today."""
    match = re.search(rf"\b(?:on|last|this past)\s+({'|'.join(_WEEKDAYS)})\b", text)
    if not match:
        return None
    days_back = (now.weekday() - _WEEKDAYS.index(match.group(1))) % 7 or 7
    return _whole_day(_day_start(now) - timedelta(days=days_back), match.group(0))


def _named_period(text: str, now: datetime) -> TimeRange | None:
    today = _day_start(now)
    monday = today - timedelta(days=now.weekday())
    first_of_month = today.replace(day=1)
    first_of_prev_month = (first_of_month - timedelta(days=1)).replace(day=1)

    # Order matters: "day before yesterday" must win over "yesterday".
    periods: list[tuple[str, Callable[[], tuple[datetime, datetime]]]] = [
        (r"day before yesterday", lambda: (today - timedelta(days=2), today - timedelta(days=1))),
        (r"last night", lambda: (today - timedelta(hours=6), today + timedelta(hours=6))),
        (r"this morning", lambda: (today, today + timedelta(hours=12))),
        (r"this afternoon", lambda: (today + timedelta(hours=12), today + timedelta(hours=17))),
        (r"this evening|tonight", lambda: (today + timedelta(hours=17), today + timedelta(days=1))),
        (r"yesterday", lambda: (today - timedelta(days=1), today)),
        (r"today", lambda: (today, now)),
        (r"this week", lambda: (monday, now)),
        (r"last week", lambda: (monday - timedelta(weeks=1), monday)),
        (r"this month", lambda: (first_of_month, now)),
        (r"last month", lambda: (first_of_prev_month, first_of_month)),
    ]
    for pattern, bounds in periods:
        if match := re.search(rf"\b(?:{pattern})\b", text):
            start, end = bounds()
            return TimeRange(start, end, match.group(0))
    return None


_RESOLVERS = [_units_ago, _rolling_window, _explicit_date, _weekday, _named_period]
