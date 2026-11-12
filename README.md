# Gemini Chat Agent

A terminal chat agent that **answers fast**, **gets things right** by grounding answers in Google Search, and
**remembers what you asked and when**, including across restarts.

- One tool: Google Search, through Gemini's built-in grounding
- Streaming answers, with sources and latency shown after every reply
- Timestamped long-term memory in SQLite: ask *"what did I ask you yesterday?"* or *"when did I ask about X?"*

Architecture and design decisions are in **[DESIGN.md](DESIGN.md)**.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your free key from https://aistudio.google.com/apikey
python app.py
```

Example session (illustrative):

```
you> who is the current CEO of Microsoft?
agent> Satya Nadella has been Microsoft's CEO since February 2014 ...
Sources:
  [1] microsoft.com - https://...
(searched: current CEO of Microsoft | first token 0.9s | total 2.1s)

you> what did I ask you yesterday?
agent> Yesterday at 3:05 PM you asked about the population of Iceland ...
(no search | first token 0.6s | total 1.0s)
```

Commands: `/new` starts a new session (memory is kept), `/history [n]` lists your recent questions with
times, `/context` shows the exact system prompt from the last turn, and `/quit` exits.

## Tests and evals

```bash
pytest                         # 41 offline tests: memory, time parsing, orchestration (no API key needed)
python -m evals.run_evals      # live evals: latency, search routing, accuracy, memory recall -> evals/results.md
```

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Free key from Google AI Studio |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Any Gemini model that supports Search grounding |
| `TIMEZONE` | system timezone | IANA name, e.g. `Asia/Kolkata`; used for "yesterday" and similar |
| `GEMINI_THINKING_BUDGET` | model default | Lower = faster, on models that support a thinking budget |
| `MEMORY_DB` | `data/memory.db` | Where memory is stored |
| `HISTORY_TURNS` | `12` | Recent messages sent verbatim each turn |
