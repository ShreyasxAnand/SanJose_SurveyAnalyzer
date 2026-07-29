"""Location layer: span collection, canonicalization guards, deterministic
matching. The one model call is faked; everything else is pure."""
import json
from collections import Counter

from app import locations


class FakeClient:
    model_id = "fake-model"

    def __init__(self, *replies):
        self.replies = list(replies)

    def complete(self, system: str, user: str) -> str:
        return self.replies.pop(0)


def test_normalize_span():
    assert locations.normalize_span("The Streets.") == "streets"
    assert locations.normalize_span("  Downtown   Area ") == "downtown area"
    assert locations.normalize_span("a park") == "park"
    assert locations.normalize_span('"St. James Park"') == "st. james park"
    # articles only stripped when something meaningful remains
    assert locations.normalize_span("the") == "the"


def test_collect_spans_counts_once_per_response():
    by_q = {
        "2": [
            {"locations": ["Downtown", "downtown"]},   # same span twice -> 1
            {"locations": ["the streets"]},
            {"locations": []},
            {},
        ],
        "3": [{"locations": ["Streets"]}],
    }
    counts, coverage = locations.collect_spans(by_q)
    assert counts == Counter({"downtown": 1, "streets": 2})
    assert coverage["2"] == {"responses": 4, "responses_with_location": 2}
    assert coverage["3"] == {"responses": 1, "responses_with_location": 1}


def test_render_spans_drops_singletons():
    counts = Counter({"streets": 5, "downtown": 3, "park by my house": 1})
    text, candidates, dropped, over_cap = locations.render_spans(counts)
    assert candidates == ["streets", "downtown"]
    assert dropped == 1
    assert over_cap == []
    assert "streets (5)" in text and "park by my house" not in text


def test_render_spans_caps_at_max_spans():
    counts = Counter({f"place {i:02d}": 10 - i for i in range(5)})
    text, candidates, dropped, over_cap = locations.render_spans(counts, max_spans=3)
    # top-3 by count kept; the rest dropped and disclosed
    assert candidates == ["place 00", "place 01", "place 02"]
    assert over_cap == ["place 03", "place 04"]
    assert dropped == 0
    assert "place 04" not in text


def test_build_locations_guards():
    reply = json.dumps({"concepts": [
        {"name": "downtown", "kind": "named",
         "spans": ["downtown", "downtown area", "invented span"]},
        {"name": "streets", "kind": "roadish",           # invalid kind
         "spans": ["streets", "downtown"]},              # downtown already used
        {"name": "empty", "kind": "type", "spans": ["not in corpus"]},
    ]})
    counts = Counter({"downtown": 5, "downtown area": 2, "streets": 9})
    locs, warnings = locations.build_locations(counts, FakeClient(reply))
    by_name = {c["name"]: c for c in locs["concepts"]}
    assert by_name["downtown"]["spans"] == ["downtown", "downtown area"]
    assert by_name["downtown"]["kind"] == "named"
    assert by_name["streets"]["kind"] == "type"          # invalid -> type
    assert by_name["streets"]["spans"] == ["streets"]    # dup span dropped
    assert "empty" not in by_name                        # no valid spans
    assert any("invented span" in w for w in warnings)
    assert any("kind" in w for w in warnings)
    assert any("no valid spans" in w for w in warnings)


def test_match_locations_sweeps_full_corpus():
    locs = {"concepts": [
        {"name": "downtown", "kind": "named", "spans": ["downtown"]},
        {"name": "streets", "kind": "type", "spans": ["street"]},
    ]}
    keys = ["k1", "k2", "k3"]
    texts = ["Downtown is dirty", "the street lights are out", "nothing here"]
    hits = locations.match_locations(locs, keys, texts)
    assert hits == {"downtown": ["k1"], "streets": ["k2"]}
    assert locations.concept_kinds(locs) == {"downtown": "named",
                                             "streets": "type"}
