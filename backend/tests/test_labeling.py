"""Labeling parser: id validation, omission handling, verbatim location guard."""
import json

from app import labeling

BATCH = [
    ("k1", "Trash everywhere near St. James Park and the bus stop"),
    ("k2", "Too many car break-ins"),
    ("k3", "Everything is fine"),
]
VALID = {"L1", "L2"}


def _raw(responses):
    return json.dumps({"responses": responses})


def test_locations_kept_verbatim_case_insensitive():
    raw = _raw([{"n": 1, "label_ids": ["L1"], "fit": 3, "sentiment": "negative",
                 "locations": ["st. james park", "bus stop"]},
                {"n": 2, "label_ids": ["L2"], "fit": 3, "sentiment": "negative"},
                {"n": 3, "label_ids": [], "uncategorized": True}])
    asg, stats = labeling.parse_label_output(raw, BATCH, VALID)
    assert asg[0]["locations"] == ["st. james park", "bus stop"]
    assert asg[1]["locations"] == []          # field absent -> empty, no error
    assert stats["invalid_locations"] == 0


def test_hallucinated_location_dropped_and_counted():
    raw = _raw([{"n": 1, "label_ids": ["L1"], "fit": 3, "sentiment": "negative",
                 "locations": ["St. James Park", "Story Road", "Downtown San Jose"]}])
    asg, stats = labeling.parse_label_output(raw, BATCH, VALID)
    # "Story Road" and "Downtown San Jose" are not substrings of k1's text
    assert asg[0]["locations"] == ["St. James Park"]
    assert stats["invalid_locations"] == 2


def test_locations_deduped_and_blank_skipped():
    raw = _raw([{"n": 1, "label_ids": ["L1"], "fit": 3, "sentiment": "negative",
                 "locations": ["bus stop", "Bus Stop", "  ", "bus stop"]}])
    asg, stats = labeling.parse_label_output(raw, BATCH, VALID)
    assert asg[0]["locations"] == ["bus stop"]
    assert stats["invalid_locations"] == 0


def test_invalid_ids_dropped_and_omitted_response_recorded():
    raw = _raw([{"n": 1, "label_ids": ["L1", "FAKE"], "fit": 3,
                 "sentiment": "negative"}])
    asg, stats = labeling.parse_label_output(raw, BATCH, VALID)
    assert asg[0]["label_ids"] == ["L1"]
    assert stats["invalid_ids"] == 1
    # responses 2 and 3 never returned -> unlabelled, flagged, empty locations
    assert stats["missing"] == 2
    for a in asg[1:]:
        assert a["not_returned"] and a["uncategorized"] and a["locations"] == []


PREFIXED = {"9_001", "9_002", "9_013", "9_063"}


def test_bare_numeric_id_recovered_to_full_id():
    """The model drops the `9_` prefix that is identical on every id in the
    taxonomy. A bare number is that question's id, not an invented one —
    dropping it recorded a coded response as "nothing fits"."""
    raw = _raw([{"n": 1, "l": ["002", "13"], "f": 3}])
    asg, stats = labeling.parse_label_output(raw, BATCH, PREFIXED)
    assert asg[0]["label_ids"] == ["9_002", "9_013"]
    assert stats["invalid_ids"] == 0
    assert not asg[0]["uncategorized"]


def test_bare_number_with_no_matching_id_is_still_invalid():
    raw = _raw([{"n": 1, "l": ["999"], "f": 3}])
    asg, stats = labeling.parse_label_output(raw, BATCH, PREFIXED)
    assert asg[0]["label_ids"] == []
    assert stats["invalid_ids"] == 1
    assert asg[0]["uncategorized"]


def test_ambiguous_bare_suffix_is_not_guessed():
    """Two ids sharing a numeric suffix make the bare form ambiguous. It must
    fail validation rather than resolve to whichever landed first."""
    raw = _raw([{"n": 1, "l": ["002"], "f": 3}])
    asg, stats = labeling.parse_label_output(raw, BATCH, {"9_002", "10_002"})
    assert asg[0]["label_ids"] == []
    assert stats["invalid_ids"] == 1


def test_recovered_and_full_forms_dedupe_together():
    raw = _raw([{"n": 1, "l": ["9_002", "002", "2"], "f": 3}])
    asg, stats = labeling.parse_label_output(raw, BATCH, PREFIXED)
    assert asg[0]["label_ids"] == ["9_002"]
    assert stats["invalid_ids"] == 0


def test_taxonomy_without_numeric_suffixes_is_unaffected():
    """No {prefix}_{digits} ids means no recovery map — bare numbers stay
    invalid, exactly as before."""
    raw = _raw([{"n": 1, "l": ["L1", "FAKE", "7"], "f": 3}])
    asg, stats = labeling.parse_label_output(raw, BATCH, VALID)
    assert asg[0]["label_ids"] == ["L1"]
    assert stats["invalid_ids"] == 2


def test_prompt_example_uses_a_real_taxonomy_id():
    """The worked example must not teach an id shape the taxonomy doesn't use.
    A `q_001` placeholder shown against `9_001` ids is what taught the model to
    drop the prefix in the first place."""
    tax = {"question_text": "q", "labels": [
        {"label_id": "9_001", "name": "n", "description": "d"},
        {"label_id": "9_002", "name": "n2", "description": "d2"}]}
    system, _ = labeling.build_label_prompts(tax, BATCH)
    example = json.loads(system[system.index('{"responses"'):].strip())
    # first worked row demonstrates MULTI-label with two real ids
    assert example["responses"][0]["l"] == ["9_001", "9_002"]
    assert "q_001" not in system


def test_bare_array_output_normalized():
    raw = json.dumps([{"n": 1, "label_ids": ["L1"], "fit": 3,
                       "sentiment": "negative"}])
    asg, _ = labeling.parse_label_output(raw, BATCH, VALID)
    assert asg[0]["label_ids"] == ["L1"]


def test_no_valid_ids_means_uncategorized():
    raw = _raw([{"n": 2, "label_ids": ["NOPE"], "fit": 2, "sentiment": "neutral"}])
    asg, _ = labeling.parse_label_output(raw, BATCH, VALID)
    assert asg[1]["uncategorized"] and asg[1]["label_ids"] == []


def test_prompt_declares_compact_keys_and_rules():
    tax = {"question_text": "q", "labels": [
        {"label_id": "L1", "name": "n", "description": "d"}]}
    system, _ = labeling.build_label_prompts(tax, BATCH)
    # the legend must name every wire key, or the abbreviations are guesswork
    for short in labeling.WIRE_KEYS.values():
        assert f"  {short} = " in system
    assert "VERBATIM" in system
    # the example must render as real JSON, not leftover .format() braces
    example = system[system.index('{"responses"'):].strip()
    parsed = json.loads(example)
    assert [r["n"] for r in parsed["responses"]] == [1, 2, 3]
    # row 1: multi-label first-hand incident with a place; row 2 teaches
    # omit-when-empty; row 3 the empty-label + f:1 (real-but-uncovered) case
    assert parsed["responses"][0]["e"] == 1 and parsed["responses"][0]["p"]
    assert "p" not in parsed["responses"][1] and "t" not in parsed["responses"][1]
    assert parsed["responses"][2]["l"] == []
    assert parsed["responses"][2]["f"] == 1


def test_compact_wire_format_parsed():
    raw = json.dumps({"responses": [
        {"n": 1, "l": ["L1"], "f": 3, "p": ["bus stop"], "a": "g", "e": 1},
        {"n": 2, "l": [], "f": 1, "a": "s", "e": 0}]})
    asg, stats = labeling.parse_label_output(raw, BATCH, VALID)
    assert asg[0]["label_ids"] == ["L1"] and asg[0]["fit"] == 3
    assert asg[0]["locations"] == ["bus stop"] and asg[0]["time_context"] == []
    assert asg[0]["actionability"] == "general" and asg[0]["event_occurred"] is True
    # empty l alone means uncategorized — no explicit flag is sent any more
    assert asg[1]["uncategorized"] and asg[1]["actionability"] == "specific"
    assert asg[1]["event_occurred"] is False
    assert stats["invalid_ids"] == 0


def test_verbose_wire_format_still_parsed():
    """The compact keys are an optimization, not a contract. A model that
    reverts to full field names must not silently yield empty labels."""
    raw = _raw([{"n": 1, "label_ids": ["L1"], "fit": 3,
                 "locations": ["bus stop"], "time_context": [],
                 "actionability": "general", "event_occurred": True}])
    asg, _ = labeling.parse_label_output(raw, BATCH, VALID)
    assert asg[0]["label_ids"] == ["L1"] and asg[0]["fit"] == 3
    assert asg[0]["locations"] == ["bus stop"]
    assert asg[0]["actionability"] == "general" and asg[0]["event_occurred"] is True


def test_verbatim_guard_still_applies_to_compact_keys():
    raw = json.dumps({"responses": [
        {"n": 1, "l": ["L1"], "f": 3,
         "p": ["St. James Park", "Story Road"], "t": ["at night"]}]})
    asg, stats = labeling.parse_label_output(raw, BATCH, VALID)
    assert asg[0]["locations"] == ["St. James Park"]   # Story Road not in text
    assert stats["invalid_locations"] == 1
    assert asg[0]["time_context"] == []                # "at night" not in text
    assert stats["invalid_time_context"] == 1


def test_explicit_uncategorized_still_honoured_if_volunteered():
    """Behaviour-identical to the verbose schema in the contradiction case:
    a volunteered flag wins over a non-empty label list, as it did before."""
    raw = json.dumps({"responses": [
        {"n": 1, "l": ["L1"], "f": 3, "uncategorized": True}]})
    asg, _ = labeling.parse_label_output(raw, BATCH, VALID)
    assert asg[0]["label_ids"] == ["L1"] and asg[0]["uncategorized"] is True


def test_event_occurred_parsed_and_coerced():
    raw = _raw([{"n": 1, "label_ids": ["L1"], "fit": 3, "sentiment": "negative",
                 "event_occurred": True},
                # the string "false" is truthy in Python — must not invert
                {"n": 2, "label_ids": ["L2"], "fit": 3, "sentiment": "negative",
                 "event_occurred": "false"},
                {"n": 3, "label_ids": ["L1"], "fit": 3, "sentiment": "neutral"}])
    asg, _ = labeling.parse_label_output(raw, BATCH, VALID)
    assert asg[0]["event_occurred"] is True
    assert asg[1]["event_occurred"] is False      # "false" string -> False
    assert asg[2]["event_occurred"] is False      # absent -> False


def test_event_denominator_excludes_uncoded_rows():
    """An unreturned row defaults to event_occurred=False; counting it in the
    denominator would understate the real event rate."""
    raw = _raw([{"n": 1, "label_ids": ["L1"], "fit": 3, "sentiment": "negative",
                 "event_occurred": True}])
    asg, stats = labeling.parse_label_output(raw, BATCH, VALID)
    assert stats["missing"] == len(BATCH) - 1
    coded = [a for a in asg if not a.get("not_returned")]
    assert len(coded) == 1
    assert sum(1 for a in asg if a["event_occurred"]) == 1


def test_time_context_verbatim_guard():
    batch = [("1:2:7", "I feel unsafe walking at night since covid started")]
    raw = _raw([{"n": 1, "label_ids": ["L1"], "fit": 3, "sentiment": "negative",
                 "time_context": ["at night", "since covid", "on weekends"]}])
    asg, stats = labeling.parse_label_output(raw, batch, VALID)
    assert asg[0]["time_context"] == ["at night", "since covid"]
    assert stats["invalid_time_context"] == 1


def test_actionability_validated():
    raw = _raw([{"n": 1, "label_ids": ["L1"], "fit": 3, "sentiment": "negative",
                 "actionability": "Specific"},
                {"n": 2, "label_ids": ["L2"], "fit": 3, "sentiment": "negative",
                 "actionability": "sorta"},
                {"n": 3, "label_ids": ["L1"], "fit": 3, "sentiment": "neutral"}])
    asg, _ = labeling.parse_label_output(raw, BATCH, VALID)
    assert asg[0]["actionability"] == "specific"   # case-normalized
    assert asg[1]["actionability"] is None         # invalid value -> None
    assert asg[2]["actionability"] is None         # absent -> None


def test_respondent_key_derived_from_response_key():
    batch = [("1:2:26", "Homeless"), ("1:2:31", "Homeless")]
    raw = _raw([{"n": 1, "label_ids": ["L1"], "fit": 3, "sentiment": "negative"}])
    asg, _ = labeling.parse_label_output(raw, batch, VALID)
    # question id stripped: same person across questions shares this key
    assert asg[0]["respondent_key"] == "1:26"
    assert asg[1]["respondent_key"] == "1:31"      # even for omitted responses
    assert labeling.respondent_key("weird-key") is None


# --- concurrency -----------------------------------------------------------
# Batches run in a thread pool, so completion order is not submission order.
# These pin the two things that must not depend on which call returned first.

class _ScrambledClient:
    """Finishes batches in reverse submission order, deterministically: each
    batch sleeps *less* the later it was submitted, so with enough workers the
    last batch returns first. `completed` records the real completion order so
    a test can prove the scramble actually happened and isn't passing vacuously.
    Optionally fails one batch, exercising failure attribution under the same
    scramble."""

    def __init__(self, n_batches, fail_on_row=None):
        self.model_id = "stub"
        self.n_batches = n_batches
        self.fail_on_row = fail_on_row
        self.completed = []

    def complete(self, system, user):
        import re, time
        rows = [int(m) for m in re.findall(r"^\d+\. response number (\d+)$", user, re.M)]
        if self.fail_on_row in rows:
            raise RuntimeError("stub failure")
        # batch index inferred from its first row; later batch -> shorter sleep
        bi = rows[0] // (len(rows) or 1)
        time.sleep(0.01 * (self.n_batches - bi))
        self.completed.append(bi)
        return json.dumps({"responses": [
            {"n": i, "label_ids": ["L1"], "fit": 3} for i in range(1, len(rows) + 1)]})


def _rows(n):
    return [(f"k{i}", f"response number {i}") for i in range(n)]


_TAX = {"question_text": "q",
        "labels": [{"label_id": "L1", "name": "n", "description": "d"}]}


def test_concurrent_batches_preserve_row_order():
    rows = _rows(12)
    client = _ScrambledClient(n_batches=4)
    asg, rep = labeling.run_labeling(
        rows, _TAX, client, batch_size=3, progress=False, workers=4)
    # the pool really did complete out of order...
    assert client.completed == [3, 2, 1, 0]
    # ...and the output is still in row order regardless
    assert [a["response_key"] for a in asg] == [k for k, _ in rows]
    assert rep["n_batches"] == 4 and rep["workers"] == 4


def test_concurrent_failed_batch_attributed_to_right_index():
    rows = _rows(12)
    # row 6 is the first row of batch index 2 (rows 6,7,8)
    asg, rep = labeling.run_labeling(
        rows, _TAX, _ScrambledClient(n_batches=4, fail_on_row=6), batch_size=3,
        progress=False, workers=4)
    assert [a["response_key"] for a in asg] == [k for k, _ in rows]
    assert [f["batch"] for f in rep["failed_batches"]] == [2]
    # exactly the failed batch's rows are marked, and no others
    assert [a["response_key"] for a in asg if a.get("batch_failed")] == ["k6", "k7", "k8"]
    assert rep["responses_not_returned"] == 3


def test_duplicate_texts_labelled_once_and_fanned_out():
    """Verbatim-identical responses cost one model slot, and every duplicate
    row gets the representative's coding under its own keys, in row order."""
    class _CountingClient:
        model_id = "stub"

        def __init__(self):
            self.rows_seen = []

        def complete(self, system, user):
            import re
            rows = re.findall(r"^\d+\. (.+)$", user, re.M)
            self.rows_seen.extend(rows)
            return json.dumps({"responses": [
                {"n": i, "l": ["L1"], "f": 3, "a": "g", "e": 0}
                for i in range(1, len(rows) + 1)]})

    rows = [("1:2:0", "Homeless"), ("1:2:1", "Dark streets"),
            ("1:2:2", "Homeless"), ("1:2:3", "Homeless"),
            ("1:2:4", "Speeding")]
    client = _CountingClient()
    asg, rep = labeling.run_labeling(rows, _TAX, client, batch_size=10,
                                     progress=False, workers=1)

    # only the 3 unique texts reached the model
    assert sorted(client.rows_seen) == ["Dark streets", "Homeless", "Speeding"]
    assert rep["duplicate_responses_collapsed"] == 2
    # output covers every row, in order, each under its own keys
    assert [a["response_key"] for a in asg] == [k for k, _ in rows]
    assert asg[2]["respondent_key"] == "1:2" and asg[3]["respondent_key"] == "1:3"
    assert all(a["label_ids"] == ["L1"] for a in asg)
    # duplicates are real copies, not shared objects
    asg[2]["label_ids"].append("mutated")
    assert asg[0]["label_ids"] == ["L1"] and asg[3]["label_ids"] == ["L1"]
    # counts cover all rows, not just unique ones
    assert rep["responses_total"] == 5
    assert rep["label_counts"]["L1"] == 5


def test_failed_batch_recovers_on_retry_pass():
    """A transient failure (fails once, succeeds on the serial retry) must
    leave no trace in failed_batches and produce real labels."""
    class _FailOnceClient(_ScrambledClient):
        def __init__(self, n_batches, fail_on_row):
            super().__init__(n_batches, fail_on_row=None)   # super never raises
            self._fail_row = fail_on_row
            self.failed_once = False

        def complete(self, system, user):
            import re
            rows = [int(m) for m in re.findall(r"^\d+\. response number (\d+)$", user, re.M)]
            if self._fail_row in rows and not self.failed_once:
                self.failed_once = True
                raise RuntimeError("transient stub failure")
            return super().complete(system, user)

    rows = _rows(12)
    asg, rep = labeling.run_labeling(
        rows, _TAX, _FailOnceClient(n_batches=4, fail_on_row=6), batch_size=3,
        progress=False, workers=4)
    assert rep["failed_batches"] == []
    assert rep["failed_batches_retried"] == 1
    assert rep["failed_batches_recovered"] == 1
    # order preserved and the recovered rows carry real labels
    assert [a["response_key"] for a in asg] == [k for k, _ in rows]
    assert all(a["label_ids"] == ["L1"] for a in asg if a["response_key"] in
               {"k6", "k7", "k8"})
    assert not any(a.get("batch_failed") for a in asg)


def test_persistent_failure_still_disclosed_after_retry():
    rows = _rows(12)
    asg, rep = labeling.run_labeling(
        rows, _TAX, _ScrambledClient(n_batches=4, fail_on_row=6), batch_size=3,
        progress=False, workers=4)
    assert [f["batch"] for f in rep["failed_batches"]] == [2]
    assert rep["failed_batches_retried"] == 1
    assert rep["failed_batches_recovered"] == 0


def test_retry_can_be_disabled():
    rows = _rows(12)
    asg, rep = labeling.run_labeling(
        rows, _TAX, _ScrambledClient(n_batches=4, fail_on_row=6), batch_size=3,
        progress=False, workers=4, retry_failed=False)
    assert [f["batch"] for f in rep["failed_batches"]] == [2]
    assert rep["failed_batches_retried"] == 0


def test_usage_add_is_atomic_under_threads():
    from concurrent.futures import ThreadPoolExecutor
    from app.llm import Usage
    u = Usage()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: u.add(10, 1), range(2000)))
    assert (u.calls, u.input_tokens, u.output_tokens) == (2000, 20000, 2000)


# --- prompt versioning -----------------------------------------------------

def test_prompt_hash_is_stable_and_description_sensitive():
    a = labeling.prompt_hash("survey of residents")
    assert a == labeling.prompt_hash("survey of residents")   # deterministic
    assert a != labeling.prompt_hash("survey of businesses")  # description versions it
    assert len(a) == 16


def test_prompt_hash_changes_when_label_prompt_changes(monkeypatch):
    """The whole point: an edit to LABEL_SYSTEM must move the hash, because a
    labeling-prompt change reshuffles per-response label sets."""
    before = labeling.prompt_hash("d")
    monkeypatch.setattr(labeling, "LABEL_SYSTEM",
                        labeling.LABEL_SYSTEM + "\n- One extra rule.")
    assert labeling.prompt_hash("d") != before


def test_labeling_and_induction_hashes_are_independent():
    """Labels runs used to be stamped with induction's hash, so an edit to
    LABEL_SYSTEM left no trace. They must not be the same value."""
    from app import induction
    assert labeling.prompt_hash("d") != induction.prompt_hash("d")
