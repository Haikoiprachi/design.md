# Gemini Chat Agent

A terminal chat agent that **answers fast**, **gets things right** by grounding answers in web search with cited sources, and
**remembers what you asked and when**, including across restarts.

- Free to run: Gemini Flash (free tier) as the model, with automatic failover when a model is overloaded
- One tool, `web_search`: Tavily (free key) or keyless public engines raced in parallel, with a cache
- Streaming answers with `[n]` citations, plus sources and latency shown after every reply
- Timestamped long-term memory in SQLite: ask *"what did I ask you yesterday?"* or *"when did I ask about X?"*

Architecture and design decisions are in **[DESIGN.md](DESIGN.md)**.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # paste your free Gemini key (https://aistudio.google.com/apikey)
                            # optional: a free Tavily key (https://tavily.com) for better search
python app.py
```

Example session (illustrative):

```
you> who is the current CEO of Microsoft?
agent> Satya Nadella is the current CEO of Microsoft [1]. He has served as CEO since February 2014 [2].
Sources:
  [1] Satya Nadella - Wikipedia - https://en.wikipedia.org/wiki/Satya_Nadella
  [2] ...
(searched: Satya Nadella Microsoft CEO | first token ... | total ...)

you> what did I ask you yesterday?
agent> Yesterday at 3:05 PM you asked about the population of Iceland ...
(no search | first token ... | total ...)
```

Commands: `/new` starts a new session (memory is kept), `/history [n]` lists your recent questions with
times, `/context` shows the exact system prompt from the last turn, and `/quit` exits.

## Tests and evals

```bash
pytest                         # 54 offline tests: memory, time parsing, search, tool loop (no API key needed)
python -m evals.run_evals      # live evals: latency, search routing, accuracy, memory recall -> evals/results.md
```

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Free key from Google AI Studio |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Any Gemini model with function calling |
| `GEMINI_FALLBACK_MODELS` | `gemini-3.6-flash,gemini-3.5-flash-lite` | Tried in order when a model is overloaded |
| `TAVILY_API_KEY` | (optional) | Free key from tavily.com; without it, keyless public engines are used |
| `TIMEZONE` | system timezone | IANA name, e.g. `Asia/Kolkata`; used for "yesterday" and similar |
| `GEMINI_THINKING_BUDGET` | model default | Lower = faster, on models that support a thinking budget |
| `MEMORY_DB` | `data/memory.db` | Where memory is stored |
| `HISTORY_TURNS` | `12` | Recent messages sent verbatim each turn |
