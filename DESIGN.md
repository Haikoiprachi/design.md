# Design

## Goal

Build the best chat agent possible with **one tool, web search**, that:

1. **Answers fast**: low time to first token, no wasted round trips.
2. **Gets things right**: fresh facts come from the web with sources, and the agent says when it doesn't know.
3. **Remembers what you asked, and when**: exact recall of past questions with timestamps, across sessions.

## Architecture

```
            ┌──────────────────────────── Orchestrator (agent/orchestrator.py) ────────────────────────────┐
 user ───►  │ 1. Build context                                                                              │
 (CLI)      │    • system prompt with the CURRENT date, time and timezone                                   │
            │    • last N turns of this session, each user turn tagged [Thu 2026-11-12 14:03]                │
            │    • MEMORY block from SQLite:                                                                 │
            │        - stats (how many messages, first one when)                                            │
            │        - if the user asks about the past: exact rows for the resolved time range,             │
            │          their first-ever messages, and their latest messages                                 │
            │        - related past messages (full-text search)                                             │
            │ 2. Save the user turn with a timestamp                                                        │
            │ 3. Stream Gemini + Google Search grounding (the only tool) ─────► tokens to the user          │
            │ 4. Save the answer and the sources it used                                                    │
            └──────────────────────────────────────────────────────────────────────────────────────────────┘
                     │                                                 │
                     ▼                                                 ▼
        Gemini API (agent/llm.py)                         SQLite + FTS5 (agent/memory.py)
        google_search grounding, streaming,               messages(id, session_id, role, content,
        retry on 429/5xx before first token               created_at UTC, sources JSON)
```

| Module | Responsibility |
|---|---|
| `agent/llm.py` | Gemini wrapper: streaming, Google Search tool, grounding metadata → sources, retries |
| `agent/memory.py` | Persistent store; recent / time-range / first / full-text queries |
| `agent/time_ranges.py` | "yesterday", "3 hours ago", "last week", "on Monday", "Nov 3" → exact `[start, end)` |
| `agent/prompts.py` | System prompt (search policy, accuracy rules, memory rules), timestamp formatting |
| `agent/orchestrator.py` | Builds context, runs a turn, records it, measures latency |
| `app.py` | Terminal UI |

## Key decisions

### 1. Memory lives in the harness, not in a tool
The brief allows one tool, and it is web search. Memory is therefore *not* a tool the model has to choose to call.
The orchestrator reads memory **before** every turn and puts what's relevant in the prompt. Benefits:
- It keeps to the one-tool rule.
- It's **faster**: no extra model round trip to call a memory tool.
- It's **more reliable**: the model can't forget to check its memory.

### 2. Time questions are answered from an exact database query
Asking an LLM to find "yesterday's" messages in a long history is error-prone. Instead, when a message is
about the conversation itself (*"did I…"*, *"what did I ask…"*, *"my first question"*), `time_ranges.py` turns
the time phrase into a concrete `[start, end)` in the user's timezone. The orchestrator then puts **every message
in that range** into the prompt with timestamps. If the range is empty it says so explicitly
(`(none: no messages in this period)`), so the model answers "no record" instead of making something up.

### 3. Every message is timestamped, and the model always knows "now"
- Stored in UTC with a fixed ISO format, so SQL string comparison matches time order. Shown in the user's
  timezone with the weekday (`Thu 2026-11-12 14:03`), which helps the model reason about "Monday" or "yesterday".
- The current date and time are in the system prompt. Without them, LLMs confidently give out-of-date answers.
- If the model copies the `[timestamp]` prefix into its reply, a small buffer strips it before streaming.

### 4. Gemini with built-in Google Search grounding
- **Free tier** through Google AI Studio.
- Search runs **on Google's side inside the same API call**: no second round trip and no separate search API key.
- Grounding metadata gives the **search queries** and **source URLs**, shown to the user and stored with the answer.
- **Gemini decides when to search.** The system prompt sets the policy: search for anything that changes
  (news, prices, roles, versions, "latest/today"), and answer stable knowledge and questions about the
  conversation directly. Skipping unneeded searches is a big latency win. The routing eval measures this.
- The default model is `gemini-3.8-flash` (fast and free), and it's configurable.

### 5. SQLite + FTS5 instead of a vector database
- No extra service and no embedding call, so retrieval is local and takes milliseconds.
- BM25 with Porter stemming handles "baking" vs "bake". Stopwords like *ask / did / yesterday* are dropped,
  so "when did I ask about sourdough" searches only for `sourdough`.
- User text is quoted before it reaches FTS, so it can't inject query syntax.
- Tradeoff: no semantic matching ("car" won't find "vehicle"). See future work.

### 6. Speed choices
| Choice | Effect |
|---|---|
| Streaming output | First words appear as soon as they're generated |
| Search only when needed | Stable questions skip the web entirely |
| Search inside the model call | No tool-call round trip |
| Local memory retrieval | Millisecond context building, no extra LLM call |
| No separate query-rewrite call | The model rewrites follow-ups itself with history in context |
| `GEMINI_THINKING_BUDGET=0` (optional) | Lower thinking budget for lower latency, where the model supports it |

### 7. Correctness choices
- The system prompt requires search-backed claims for anything time-sensitive, a preference for primary sources,
  "as of <date>" phrasing, and admitting uncertainty instead of guessing.
- Sources are always shown, so the user can check them.
- Memory claims are limited to what's in the context; empty periods are stated explicitly.

## Failure modes

| Failure | Handling |
|---|---|
| Free-tier rate limit (429) or 5xx | Exponential backoff with up to 3 retries, only **before** any text has streamed (avoids duplicate output); then a clear message, and the chat continues |
| Search returns poor or conflicting sources | The prompt tells the model to say so and show sources |
| Model echoes the timestamp prefix | Stripped by the orchestrator |
| Vague time ("an hour ago") | Generous window; the model sees exact timestamps and narrows it down |
| Very long history | Only the last `HISTORY_TURNS` messages go in verbatim; older ones come in via time-range or full-text retrieval |
| Network or API error | The turn fails gracefully; the user message stays in memory |

## Evaluation

`pytest` runs **41 offline tests** with no API key, covering time-phrase parsing, memory storage and retrieval,
and orchestration with a fake LLM (prompt contents, timestamps, empty-period handling, prefix stripping,
persistence across sessions).

`python -m evals.run_evals` runs live evals and writes `evals/results.md`:
- **Freshness** (`evals/freshness.jsonl`): stable and time-sensitive questions. Measures answer accuracy
  (keyword checks), **search routing accuracy** (did it search exactly when it should), and time to first
  token and total latency (p50/p95).
- **Memory** (`evals/memory.jsonl`): seeds past conversations at fixed times in the past, then asks
  "what did I ask yesterday?", "when did I ask about sourdough?", "what was my first question?", "what's my name?"
  (cross-session), and a no-record case that checks for hallucination.

Results: see [`evals/results.md`](evals/results.md) once generated.

## Tradeoffs and limitations
- **Free tier**: rate limits apply, and Google may use free-tier prompts to improve its products.
  A paid key removes both; the code doesn't change.
- **Less control over searching**: Gemini grounding doesn't allow forcing or blocking a search per query;
  it is steered by the prompt and checked by the routing eval.
- **Single user**: one local database per install, with no auth or multi-user isolation.
- **Keyword retrieval**: FTS has no semantic matching.

## Future work
1. Add embeddings (e.g. Gemini embeddings or sqlite-vec) alongside FTS for semantic recall.
2. Periodically summarize old sessions into a short **user profile** (name, preferences) that's always in context.
3. Use context caching for the static system prompt to cut latency and cost.
4. Add a web UI with server-sent events streaming, using the same orchestrator.
5. Use an LLM judge for open-ended accuracy evals instead of keyword checks.
