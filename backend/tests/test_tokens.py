"""Prompt-token counting: exactness where it is affordable, honesty where it
is not.

Everything here runs against a fake SDK client. The autouse conftest fixture
disables counting for the rest of the suite; these tests re-enable it against
a stub so the counting path is actually exercised without a network call.
"""
import pytest

from app import tokens


class _FakeResponse:
    def __init__(self, total):
        self.total_tokens = total


class _FakeModels:
    """Counts tokens at a fixed, per-kind rate so assertions can be exact.

    Instruction text and survey prose really do tokenize differently, and the
    counter pools them separately for that reason — so the fake has to model
    it or the pooling logic is untested.
    """

    SYSTEM_CHARS_PER_TOKEN = 4.0
    USER_CHARS_PER_TOKEN = 5.0

    def __init__(self):
        self.calls = []
        self.fail_after = None

    def count_tokens(self, *, model, contents, config=None):
        self.calls.append((model, contents, config))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("endpoint said no")
        system = getattr(config, "system_instruction", None) or ""
        if system:
            return _FakeResponse(round(len(system) / self.SYSTEM_CHARS_PER_TOKEN))
        return _FakeResponse(round(len(contents) / self.USER_CHARS_PER_TOKEN))


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


@pytest.fixture()
def counter(monkeypatch):
    """A counter wired to the fake client, with the real call budget.

    The stub honours `_unavailable` exactly as the real `_get_client` does —
    that flag is what stops a dead endpoint being retried once per prompt, so
    a fake that ignored it would make the give-up behaviour untestable.
    """
    fake = _FakeClient()
    monkeypatch.setattr(tokens.TokenCounter, "_get_client",
                        lambda self: None if self._unavailable else fake)
    made = tokens.TokenCounter("fake-model", max_calls=24)
    made.fake = fake
    return made


SYSTEM = "S" * 400
def _users(n, size=500):
    return [f"{i}" + "u" * (size - 1) for i in range(n)]


# --- the exact path --------------------------------------------------------


def test_a_small_stage_is_counted_exactly(counter):
    users = _users(2)
    result = counter.estimate(SYSTEM, users)

    assert result.basis == tokens.EXACT
    # system counted once and charged per prompt, plus each user block
    expected = round(400 / 4) * 2 + 2 * round(500 / 5)
    assert result.tokens == expected


def test_no_prompts_costs_nothing(counter):
    result = counter.estimate(SYSTEM, [])
    assert result.tokens == 0
    assert result.calls == 0
    assert counter.fake.models.calls == []


def test_the_system_prompt_is_sent_as_system_instruction(counter):
    counter.estimate(SYSTEM, _users(1))
    systems = [getattr(c[2], "system_instruction", None)
               for c in counter.fake.models.calls]
    # complete() sends it that way, so a count that left it out of the
    # system_instruction slot would be counting a different prompt
    assert SYSTEM in systems


# --- the sampled path ------------------------------------------------------


def test_a_large_stage_is_sampled_and_scaled(counter):
    users = _users(50)
    result = counter.estimate(SYSTEM, users)

    assert result.basis == tokens.SAMPLED
    # the whole stage is priced without 50 round-trips
    assert result.calls <= tokens.FIRST_STAGE_CALLS
    exact = round(400 / 4) * 50 + 50 * round(500 / 5)
    # the fake tokenizes at a fixed rate, so a correct ratio reproduces the
    # exact figure to within rounding
    assert result.tokens == pytest.approx(exact, rel=0.02)


def test_the_sample_is_spread_not_taken_from_the_front(counter):
    # blocks get longer down the list; sampling the head would understate the
    # whole stage, which is exactly what the old first-batch heuristic did
    users = [("u" * (100 * (i + 1))) for i in range(20)]
    result = counter.estimate(SYSTEM, users)

    sampled = [c[1] for c in counter.fake.models.calls if "u" in c[1]]
    assert len(sampled) >= 2
    assert len({len(s) for s in sampled}) > 1, "every sample was the same size"
    exact = round(400 / 4) * 20 + sum(round(len(u) / 5) for u in users)
    assert result.tokens == pytest.approx(exact, rel=0.1)


def test_measurements_pool_across_stages(counter):
    """The fix that matters on a multi-question dataset.

    A per-stage budget spent first-come leaves later questions on the chars/4
    fallback. Pooling the ratio means the first question's measurements price
    the last one.
    """
    for _ in range(20):
        result = counter.estimate(SYSTEM, _users(40))
        assert result.basis == tokens.SAMPLED, "a later stage fell back"
    assert counter.calls_made <= counter.max_calls


def test_later_stages_cost_fewer_calls_than_the_first(counter):
    first = counter.estimate(SYSTEM, _users(40))
    second = counter.estimate(SYSTEM + "x", _users(40))
    assert first.calls == tokens.FIRST_STAGE_CALLS
    assert second.calls <= tokens.LATER_STAGE_CALLS < first.calls


def test_the_budget_is_never_exceeded(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(tokens.TokenCounter, "_get_client", lambda self: fake)
    counter = tokens.TokenCounter("fake-model", max_calls=3)
    for i in range(10):
        counter.estimate(SYSTEM + str(i), _users(30))
    assert counter.calls_made <= 3


# --- degradation -----------------------------------------------------------


def test_no_client_falls_back_to_the_character_heuristic(monkeypatch):
    monkeypatch.setattr(tokens.TokenCounter, "_get_client", lambda self: None)
    counter = tokens.TokenCounter("fake-model")
    users = _users(10)

    result = counter.estimate(SYSTEM, users)

    assert result.basis == tokens.HEURISTIC
    total_chars = len(SYSTEM) * len(users) + sum(len(u) for u in users)
    assert result.tokens == round(total_chars * tokens.HEURISTIC_TOKENS_PER_CHAR)


def test_a_failing_endpoint_is_not_retried_to_death(counter):
    counter.fake.models.fail_after = 0
    result = counter.estimate(SYSTEM, _users(30))
    assert result.basis == tokens.HEURISTIC
    # one failure is the whole endpoint (auth, region, model id) — trying
    # again just burns the budget and the analyst's patience
    assert len(counter.fake.models.calls) == 1


def test_a_failure_partway_through_still_yields_a_measured_estimate(counter):
    counter.estimate(SYSTEM, _users(30))          # fills the pool
    counter.fake.models.fail_after = len(counter.fake.models.calls)
    result = counter.estimate(SYSTEM + "different", _users(30))
    # the pool survives the failure, so this is still measured rather than
    # dropping the whole stage back to chars/4
    assert result.basis == tokens.SAMPLED


def test_repeated_texts_are_memoised_not_recounted(counter):
    counter.estimate(SYSTEM, _users(2))
    before = len(counter.fake.models.calls)
    counter.estimate(SYSTEM, _users(2))
    assert len(counter.fake.models.calls) == before


def test_a_zero_or_negative_count_is_treated_as_a_failure(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(fake.models, "count_tokens",
                        lambda **kw: _FakeResponse(0))
    monkeypatch.setattr(tokens.TokenCounter, "_get_client", lambda self: fake)
    counter = tokens.TokenCounter("fake-model")
    # a zero from the endpoint is a malformed answer, not a free prompt
    assert counter.estimate(SYSTEM, _users(4)).basis == tokens.HEURISTIC


def test_spread_indices_are_spread(monkeypatch):
    assert tokens._spread_indices(10, 3) == [0, 3, 6]
    assert tokens._spread_indices(3, 10) == [0, 1, 2]
    assert tokens._spread_indices(0, 3) == []
    assert tokens._spread_indices(10, 0) == []
