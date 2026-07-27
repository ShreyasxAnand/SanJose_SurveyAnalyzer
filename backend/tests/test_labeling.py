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


def test_bare_array_output_normalized():
    raw = json.dumps([{"n": 1, "label_ids": ["L1"], "fit": 3,
                       "sentiment": "negative"}])
    asg, _ = labeling.parse_label_output(raw, BATCH, VALID)
    assert asg[0]["label_ids"] == ["L1"]


def test_no_valid_ids_means_uncategorized():
    raw = _raw([{"n": 2, "label_ids": ["NOPE"], "fit": 2, "sentiment": "neutral"}])
    asg, _ = labeling.parse_label_output(raw, BATCH, VALID)
    assert asg[1]["uncategorized"] and asg[1]["label_ids"] == []


def test_prompt_mentions_locations_rule():
    tax = {"question_text": "q", "labels": [
        {"label_id": "L1", "name": "n", "description": "d"}]}
    system, _ = labeling.build_label_prompts(tax, BATCH)
    assert "VERBATIM" in system and '"locations"' in system
    assert '"time_context"' in system and '"actionability"' in system


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
