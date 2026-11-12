"""Settings loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    db_path: Path
    timezone: tzinfo
    thinking_budget: int | None
    history_turns: int


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
            "and put it in .env (see .env.example)."
        )

    tz_name = os.getenv("TIMEZONE", "").strip()
    timezone = ZoneInfo(tz_name) if tz_name else datetime.now().astimezone().tzinfo

    budget = os.getenv("GEMINI_THINKING_BUDGET", "").strip()
    return Settings(
        api_key=api_key,
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip(),
        db_path=Path(os.getenv("MEMORY_DB", PROJECT_ROOT / "data" / "memory.db")),
        timezone=timezone,
        thinking_budget=int(budget) if budget else None,
        history_turns=int(os.getenv("HISTORY_TURNS", "12")),
    )
