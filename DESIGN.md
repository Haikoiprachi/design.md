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
            │ 3. Stream Gemini; if it calls web_search (the only tool), run the queries in parallel,        │
            │    send back numbered results, keep streaming ─────► tokens + [n] citations to the user       │
            │ 4. Save the answer and the sources it used                                                    │
            └──────────────────────────────────────────────────────────────────────────────────────────────┘
                     │                          │                              │
                     ▼                          ▼                              ▼
        Gemini API (agent/llm.py)     web_search (agent/search.py)     SQLite + FTS5 (agent/memory.py)
        streaming, function calling,  Tavily (free key) or keyless     messages(id, session_id, role,
        fail over to the next model   engines raced in parallel,       content, created_at UTC,
        on 503/504/timeouts           10-minute result cache           sources JSON)
```

| Module | Responsibility |
|---|---|
| `agent/llm.py` | Gemini wrapper: streaming, the `web_search` function-calling loop, citation numbering, model failover |
| `agent/search.py` | The one tool: Tavily or keyless engines raced in parallel, plus a result cache |
| `agent/memory.py` | Persistent store; recent / time-range / first / full-text queries |
| `agent/time_ranges.py` | "yesterday", "3 hours ago", "last week", "on Monday", "Nov 3" → exact `[start, end)` |
| `agent/prompts.py` | System prompt (search policy, accuracy rules, memory rules), timestamp formatting |
| `agent/orchestrator.py` | Builds context, runs a turn, records it, measures latency |
| `web.py` + `static/index.html` | Web UI: FastAPI server streaming Server-Sent Events to a single dependency-free page |
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

### 4. Gemini (free tier) with our own `web_search` function
- **The model is Gemini Flash on the free tier.** The default is `gemini-3.5-flash`, which was the fastest
  available model in testing.
- **Why not Gemini's built-in Google Search grounding?** It was the first design, but on the free tier it returned
  `429 RESOURCE_EXHAUSTED` on every model, while plain generation worked. So search is a normal function tool
  we run ourselves. It's still one tool, and it brings advantages:
  - We control the provider, timeouts, caching and parallelism.
  - Results are **numbered across the whole turn**, so the model's `[n]` citations match the sources list exactly.
  - Search failures go back to the model as data, so it can say "I couldn't verify this" instead of crashing.
- **Search providers** (`agent/search.py`):
  - **Tavily** when `TAVILY_API_KEY` is set (free tier, built for LLM agents, fast and reliable).
  - Otherwise **keyless public engines** (Yahoo, DuckDuckGo, Bing via `ddgs`), queried **in parallel**, taking the
    first non-empty result. In testing, single engines failed often (DuckDuckGo 0/4, Bing 2/4, Yahoo 3/4 queries),
    so racing them gives both better reliability and the latency of the fastest engine.
  - A 10-minute cache makes repeated or follow-up queries free.
- **Gemini decides when to search.** The system prompt sets the policy: search for anything that changes
  (news, prices, roles, versions, "latest/today"), and answer stable knowledge and questions about the
  conversation directly. The model can issue several searches at once (run in parallel), at most 3 rounds,
  and the final round disables tools so it always ends with an answer.
- **Model failover.** Free-tier models are often overloaded (503) or slow (504/timeouts); in testing, 3.8, 3.7
  and 3.1-lite all timed out while 3.5 Flash answered in about 2 seconds. On an overload before any text has
  streamed, the agent **switches to the next model in `GEMINI_FALLBACK_MODELS` immediately**, and keeps using
  whichever model last worked. It only backs off after every model has failed.

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
| Parallel searches + engine racing + cache | Search costs about one fastest-engine round trip, or nothing if cached |
| Immediate model failover | An overloaded model costs one failed call, not a long wait |
| Local memory retrieval | Millisecond context building, no extra LLM call |
| No separate query-rewrite call | The model rewrites follow-ups itself with history in context |
| `GEMINI_THINKING_BUDGET=0` (optional) | Lower thinking budget for lower latency, where the model supports it |

### 7. Web UI
- `web.py` is a thin FastAPI layer over the same `ChatAgent`; nothing about the agent changes for the web.
- `POST /api/chat` streams **Server-Sent Events**: `session` (its id), `search` (queries, shown live as
  "Searching the web: …"), `delta` (text), `done` (answer, sources, queries, latency, which model answered), or `error`.
- The server is stateless per request: the browser keeps the `session_id`, and the conversation itself lives in SQLite.
  `GET /api/history` feeds the sidebar (every question, grouped by day with times, which makes "remembers what you
  asked and when" visible); `GET /api/sessions/{id}` reopens any past conversation so you can continue it.
- `static/index.html` is one file with no build step and no external requests. It renders a safe subset of
  Markdown itself (HTML is escaped first), turns `[n]` into links to source n, and lists only cited sources.
- The SQLite connection is shared across the server's worker threads behind a lock.

### 8. Correctness choices
- The system prompt requires search-backed claims for anything time-sensitive, a preference for primary sources,
  "as of <date>" phrasing, and admitting uncertainty instead of guessing.
- Sources are always shown, so the user can check them.
- Memory claims are limited to what's in the context; empty periods are stated explicitly.

## Failure modes

| Failure | Handling |
|---|---|
| Model overloaded (503/504) or timed out | Fail over to the next model immediately; back off only after every model has failed; only **before** any text has streamed (avoids duplicate output) |
| Free-tier rate limit (429) | Same retry path, then a clear "wait a minute" message, and the chat continues |
| All search engines fail | The error goes back to the model as the tool result; it answers from its own knowledge and says so |
| Model ends a turn with no text | Asked once more with tools disabled |
| Search returns poor or conflicting sources | The prompt tells the model to say so and show sources |
| Model echoes the timestamp prefix | Stripped by the orchestrator |
| Vague time ("an hour ago") | Generous window; the model sees exact timestamps and narrows it down |
| Very long history | Only the last `HISTORY_TURNS` messages go in verbatim; older ones come in via time-range or full-text retrieval |
| Network or API error | The turn fails gracefully; the user message stays in memory |

## Evaluation

`pytest` runs **54 offline tests** with no API key, covering time-phrase parsing, memory storage and retrieval,
orchestration with a fake LLM (prompt contents, timestamps, empty-period handling, prefix stripping,
persistence across sessions), the search providers (fallthrough, racing, cache), and the Gemini tool loop with
a fake client speaking real SDK types (citation numbering, parallel calls, round cap, empty-answer retry, failover).

`python -m evals.run_evals` runs live evals and writes `evals/results.md`:
- **Freshness** (`evals/freshness.jsonl`): stable and time-sensitive questions. Measures answer accuracy
  (keyword checks), **search routing accuracy** (did it search exactly when it should), and time to first
  token and total latency (p50/p95).
- **Memory** (`evals/memory.jsonl`): seeds past conversations at fixed times in the past, then asks
  "what did I ask yesterday?", "when did I ask about sourdough?", "what was my first question?", "what's my name?"
  (cross-session), and a no-record case that checks for hallucination.

### Latest results ([`evals/results.md`](evals/results.md), `gemini-3.5-flash`, keyless search)

| Suite | Accuracy | Search routing | Time to first token p50 / p95 |
|---|---|---|---|
| Freshness (11) | 11/11 | **11/11** (searched for all 6 fresh, none of the 5 stable) | 6.1s / 21.8s |
| Memory (6) | **6/6** | never searched (correct) | 1.2s / 2.0s |

Honest reading:
- **Memory is fast and exact**: around 1.2 seconds, with correct days and times, cross-session recall, and a correct "no, you didn't".
- **Stable questions** answer in about 1.2 to 2 seconds with no search. The 11.9s outlier was the first call of the
  run, hitting a cold or overloaded free-tier model.
- **Searched answers take 4 to 10 seconds, with a long tail.** The cost is two model calls plus keyless search.
  This is the main thing a Tavily key and a paid tier would improve.
- **Keyword checks are lenient.** `fresh-news` passed on routing, but the answer said it couldn't fetch live
  headlines, because snippets from keyless search lacked them. Several fresh cases have no keyword to check
  (the answers change daily). Next step: an LLM judge with the retrieved sources (see future work).
- The pre-fix run found a real bug: "the **very** first question I **ever** asked" wasn't recognised as a
  memory question. It's fixed and covered by a regression test.

## Tradeoffs and limitations
- **Free tier**: the quota is tiny (measured: **20 requests per model per day**, and one searched answer uses 2 or 3
  requests), models get overloaded at busy times (latency varies a lot), and Google may use free-tier prompts to
  improve its products. Failover across 4 Flash models stretches the daily budget; a paid key removes all of these
  limits with no code change.
- **Keyless search is best-effort**: public engines block scrapers unpredictably and return snippets, not pages,
  so live numbers (weather, prices) are sometimes missing. A free Tavily key is the recommended setup.
- **Searching takes an extra round trip** compared with built-in grounding, but only on turns that need it.
- **Single user**: one local database per install, with no auth or multi-user isolation.
- **Keyword retrieval**: FTS has no semantic matching.

## Future work
1. Add embeddings (e.g. Gemini embeddings or sqlite-vec) alongside FTS for semantic recall.
2. Periodically summarize old sessions into a short **user profile** (name, preferences) that's always in context.
3. Use context caching for the static system prompt to cut latency and cost.
4. Fetch and extract the top result page when snippets lack the needed number (weather, prices).
5. Show search progress and source previews inline in the web UI as results arrive.
6. Use an LLM judge for open-ended accuracy evals instead of keyword checks.
