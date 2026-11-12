"""Live evals against the Gemini API: latency, search routing, answer accuracy, memory recall.

Usage:  python -m evals.run_evals [--delay 4] [--only freshness|memory]
Writes a Markdown report to evals/results.md.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from agent.config import load_settings
from agent.llm import GeminiLLM, RateLimitedError
from agent.memory import MemoryStore
from agent.orchestrator import ChatAgent, TurnResult
from agent.search import build_search

EVAL_DIR = Path(__file__).resolve().parent


@dataclass
class CaseResult:
    suite: str
    case_id: str
    passed: bool
    routing_ok: bool | None
    searched: bool
    ttft_s: float | None
    total_s: float
    answer: str
    error: str = ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between calls (free-tier limits)")
    parser.add_argument("--only", choices=["freshness", "memory"])
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    settings = load_settings()
    search = build_search(settings.tavily_api_key)
    llm = GeminiLLM(
        settings.api_key, settings.model, search,
        fallback_models=settings.fallback_models, thinking_budget=settings.thinking_budget,
    )
    print(f"model={settings.model} search={search.name}")
    results: list[CaseResult] = []

    suites = [("freshness", run_freshness_case), ("memory", run_memory_case)]
    for suite, runner in suites:
        if args.only and args.only != suite:
            continue
        for case in load_cases(EVAL_DIR / f"{suite}.jsonl"):
            result = runner(case, llm, settings.timezone)
            results.append(result)
            mark = "PASS" if result.passed else "FAIL"
            print(f"[{mark}] {suite}/{result.case_id}  searched={result.searched}  "
                  f"total={result.total_s:.2f}s  {result.error}")
            time.sleep(args.delay)

    report = render_report(results, settings.model)
    (EVAL_DIR / "results.md").write_text(report, encoding="utf-8")
    print("\n" + report)


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ask(agent: ChatAgent, question: str) -> TurnResult:
    result = None
    for event in agent.chat(question):
        if isinstance(event, TurnResult):
            result = event
    assert result is not None
    return result


def run_freshness_case(case: dict, llm: GeminiLLM, tz) -> CaseResult:
    agent = ChatAgent(MemoryStore(), llm, tz=tz)
    try:
        result = ask(agent, case["question"])
    except RateLimitedError as exc:
        return CaseResult("freshness", case["id"], False, None, False, None, 0.0, "", str(exc))
    keywords_ok = all(k.lower() in result.answer.lower() for k in case.get("expect", []))
    routing_ok = result.searched == case["must_search"]
    return CaseResult("freshness", case["id"], keywords_ok and bool(result.answer), routing_ok,
                      result.searched, result.ttft_s, result.total_s, result.answer)


def run_memory_case(case: dict, llm: GeminiLLM, tz) -> CaseResult:
    memory = MemoryStore()
    now = datetime.now(tz)
    seeded = [
        memory.add("earlier-session", s["role"], s["content"],
                   created_at=now - timedelta(minutes=s["minutes_ago"]))
        for s in case["seed"]
    ]
    agent = ChatAgent(memory, llm, tz=tz)
    try:
        result = ask(agent, case["question"])
    except RateLimitedError as exc:
        return CaseResult("memory", case["id"], False, None, False, None, 0.0, "", str(exc))

    answer = result.answer.lower()
    passed = all(k.lower() in answer for k in case.get("expect", []))
    if "expect_any" in case:
        passed = passed and any(k.lower() in answer for k in case["expect_any"])
    if "expect_date_of_seed" in case:
        passed = passed and mentions_date(result.answer, seeded[case["expect_date_of_seed"]].created_at.astimezone(tz))
    return CaseResult("memory", case["id"], passed, None, result.searched,
                      result.ttft_s, result.total_s, result.answer)


def mentions_date(answer: str, when: datetime) -> bool:
    """Accept any common way of naming the day: ISO date, 'Nov 9', '9 November', or the weekday."""
    text = answer.lower()
    month_named = when.strftime("%b").lower() in text and bool(re.search(rf"\b{when.day}(?:st|nd|rd|th)?\b", text))
    return when.strftime("%Y-%m-%d") in text or month_named or when.strftime("%A").lower() in text


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def render_report(results: list[CaseResult], model: str) -> str:
    lines = [f"# Eval results\n\nModel: `{model}` · run {datetime.now():%Y-%m-%d %H:%M}\n"]
    for suite in ("freshness", "memory"):
        rows = [r for r in results if r.suite == suite]
        if not rows:
            continue
        passed = sum(r.passed for r in rows)
        lines.append(f"## {suite.title()}\n\nAnswer accuracy: **{passed}/{len(rows)}**")
        routed = [r for r in rows if r.routing_ok is not None]
        if routed:
            lines.append(f"Search routing accuracy: **{sum(r.routing_ok for r in routed)}/{len(routed)}**")
        ttfts = [r.ttft_s for r in rows if r.ttft_s is not None]
        totals = [r.total_s for r in rows if r.total_s]
        if ttfts:
            lines.append(f"Time to first token: p50 **{statistics.median(ttfts):.2f}s**, "
                         f"p95 **{percentile(ttfts, 95):.2f}s**")
        if totals:
            lines.append(f"Total time: p50 **{statistics.median(totals):.2f}s**, "
                         f"p95 **{percentile(totals, 95):.2f}s**")
        lines.append("\n| case | pass | searched | ttft | total | answer |\n|---|---|---|---|---|---|")
        for r in rows:
            ttft = f"{r.ttft_s:.2f}s" if r.ttft_s is not None else "-"
            answer = (r.error or r.answer).replace("\n", " ").replace("|", "\\|")[:120]
            lines.append(f"| {r.case_id} | {'✅' if r.passed else '❌'} | {r.searched} | {ttft} "
                         f"| {r.total_s:.2f}s | {answer} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
