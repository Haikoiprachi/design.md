"""GeminiLLM's tool loop and failover, driven by a fake client that speaks real SDK types."""

import pytest
from google.genai import errors, types

from agent.llm import ChatTurn, GeminiLLM, Grounding, RateLimitedError, SearchStarted, TextChunk
from agent.search import SearchError, SearchResult


def chunk(*parts):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))]
    )


def text(value):
    return types.Part(text=value)


def search_call(query, call_id="c1"):
    return types.Part(function_call=types.FunctionCall(id=call_id, name="web_search", args={"query": query}))


class FakeModels:
    """Each generate_content_stream call pops the next scripted response (list of chunks or an exception)."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def generate_content_stream(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        yield from step


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


class FakeSearch:
    name = "fake"

    def __init__(self, results=None, fail=False):
        self.results = results or {}
        self.fail = fail
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        if self.fail:
            raise SearchError("all engines down")
        return self.results.get(query, [])


def make_llm(script, search=None, **kwargs):
    client = FakeClient(script)
    llm = GeminiLLM("key", "main-model", search or FakeSearch(), client=client, **kwargs)
    return llm, client


def run(llm):
    return list(llm.stream("system", [ChatTurn("user", "hi")]))


def api_error(code):
    return errors.APIError(code, {"error": {"code": code, "message": "x", "status": "X"}})


def test_plain_answer_streams_without_grounding():
    llm, client = make_llm([[chunk(text("Hel")), chunk(text("lo"))]])
    assert run(llm) == [TextChunk("Hel"), TextChunk("lo")]
    assert len(client.models.requests) == 1


def test_search_round_trip_numbers_sources_and_dedupes_urls():
    search = FakeSearch({
        "ceo microsoft": [SearchResult("Wiki", "https://w", "Satya Nadella"),
                          SearchResult("MS", "https://ms", "CEO page")],
        "ms ceo since": [SearchResult("MS", "https://ms", "since 2014")],
    })
    script = [
        [chunk(search_call("ceo microsoft", "a"), search_call("ms ceo since", "b"))],
        [chunk(text("Satya Nadella [1][2]."))],
    ]
    llm, client = make_llm(script, search)

    events = run(llm)

    assert events[:-1] == [SearchStarted(("ceo microsoft", "ms ceo since")), TextChunk("Satya Nadella [1][2].")]
    grounding = events[-1]
    assert isinstance(grounding, Grounding)
    assert [s.uri for s in grounding.sources] == ["https://w", "https://ms"]
    assert grounding.queries == ("ceo microsoft", "ms ceo since")

    second_request = client.models.requests[1]["contents"]
    assert second_request[-2].role == "model" and second_request[-2].parts[0].function_call.id == "a"
    responses = [p.function_response for p in second_request[-1].parts]
    assert [r.id for r in responses] == ["a", "b"]
    assert [r["id"] for r in responses[0].response["results"]] == [1, 2]
    assert [r["id"] for r in responses[1].response["results"]] == [2]  # same URL, same citation number


def test_search_failure_is_reported_to_the_model_not_raised():
    script = [[chunk(search_call("x"))], [chunk(text("I couldn't verify that."))]]
    llm, client = make_llm(script, FakeSearch(fail=True))
    events = run(llm)
    assert events == [SearchStarted(("x",)), TextChunk("I couldn't verify that.")]
    payload = client.models.requests[1]["contents"][-1].parts[0].function_response.response
    assert "all engines down" in payload["error"]


def test_search_rounds_are_capped_and_last_round_disables_tools():
    script = [[chunk(search_call("q1"))], [chunk(search_call("q2"))], [chunk(text("done"))]]
    llm, client = make_llm(script, max_search_rounds=2)
    run(llm)
    modes = [r["config"].tool_config.function_calling_config.mode for r in client.models.requests]
    assert modes == ["AUTO", "AUTO", "NONE"]


def test_empty_answer_is_retried_once_without_tools():
    script = [[chunk(search_call("q"))], [chunk(text(""))], [chunk(text("Answer."))]]
    llm, client = make_llm(script)
    assert run(llm)[:2] == [SearchStarted(("q",)), TextChunk("Answer.")]
    modes = [r["config"].tool_config.function_calling_config.mode for r in client.models.requests]
    assert modes == ["AUTO", "AUTO", "NONE"]


def test_overloaded_model_fails_over_immediately_and_sticks():
    script = [api_error(503), [chunk(text("from fallback"))], [chunk(text("again"))]]
    llm, client = make_llm(script, fallback_models=("backup-model",))
    assert run(llm) == [TextChunk("from fallback")]
    run(llm)
    assert [r["model"] for r in client.models.requests] == ["main-model", "backup-model", "backup-model"]


def test_network_timeout_also_fails_over():
    import httpx

    llm, client = make_llm([httpx.ReadTimeout("slow"), [chunk(text("ok"))]], fallback_models=("backup",))
    assert run(llm) == [TextChunk("ok")]


def test_rate_limit_after_all_retries_raises_friendly_error(monkeypatch):
    monkeypatch.setattr("agent.llm.time.sleep", lambda s: None)
    llm, _ = make_llm([api_error(429)] * 2, max_retries=1)
    with pytest.raises(RateLimitedError):
        run(llm)


def test_non_retryable_errors_propagate():
    llm, _ = make_llm([api_error(400)], fallback_models=("backup",))
    with pytest.raises(errors.APIError):
        run(llm)
