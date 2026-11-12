"""Terminal chat UI: streams answers, shows sources and latency, keeps memory across runs."""

from __future__ import annotations

import sys

from agent.config import load_settings
from agent.llm import GeminiLLM, RateLimitedError
from agent.memory import MemoryStore
from agent.orchestrator import ChatAgent, TextDelta, TurnResult
from agent.prompts import format_timestamp, snippet
from agent.search import build_search

DIM, BOLD, CYAN, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[0m"
HELP = f"""{DIM}Commands: /new  start a new session (memory is kept)
          /history [n]  show your last n questions with times
          /context  show the system prompt sent on the last turn
          /quit{RESET}"""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = load_settings()
    memory = MemoryStore(settings.db_path)
    search = build_search(settings.tavily_api_key)
    llm = GeminiLLM(
        settings.api_key, settings.model, search,
        fallback_models=settings.fallback_models, thinking_budget=settings.thinking_budget,
    )
    agent = ChatAgent(memory, llm, tz=settings.timezone, history_turns=settings.history_turns)

    print(f"{BOLD}Gemini chat agent{RESET} {DIM}({settings.model}, search: {search.name}, "
          f"memory: {settings.db_path}){RESET}")
    print(HELP)
    while True:
        try:
            user_text = input(f"\n{BOLD}{CYAN}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.startswith("/"):
            if not handle_command(user_text, agent, memory, settings.timezone):
                break
            continue
        run_turn(agent, user_text)
    memory.close()


def handle_command(command: str, agent: ChatAgent, memory: MemoryStore, tz) -> bool:
    name, _, arg = command.partition(" ")
    if name in ("/quit", "/exit"):
        return False
    if name == "/new":
        agent.new_session()
        print(f"{DIM}New session started. Earlier conversations are still remembered.{RESET}")
    elif name == "/history":
        count = int(arg) if arg.strip().isdigit() else 10
        for message in memory.latest_user_messages(count):
            print(f"{DIM}[{format_timestamp(message.created_at, tz)}]{RESET} {snippet(message.content, 100)}")
    elif name == "/context":
        print(f"{DIM}{agent.last_system_prompt or '(no turns yet)'}{RESET}")
    else:
        print(HELP)
    return True


def run_turn(agent: ChatAgent, user_text: str) -> None:
    print(f"{BOLD}agent>{RESET} ", end="", flush=True)
    try:
        for event in agent.chat(user_text):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, TurnResult):
                print_footer(event)
    except RateLimitedError as exc:
        print(f"\n{DIM}{exc}{RESET}")
    except Exception as exc:  # keep the chat alive on network/API errors
        print(f"\n{DIM}Error: {exc}{RESET}")


def print_footer(result: TurnResult) -> None:
    print()
    numbered = list(enumerate(result.sources, start=1))
    cited = [(i, s) for i, s in numbered if f"[{i}]" in result.answer] or numbered[:5]
    if cited:
        print(f"{DIM}Sources:")
        for i, source in cited:
            print(f"  [{i}] {source.title} - {source.uri}")
        print(RESET, end="")
    searched = f"searched: {', '.join(result.search_queries)}" if result.search_queries else "no search"
    ttft = f"{result.ttft_s:.2f}s" if result.ttft_s is not None else "n/a"
    print(f"{DIM}({searched} | first token {ttft} | total {result.total_s:.2f}s){RESET}")


if __name__ == "__main__":
    main()
