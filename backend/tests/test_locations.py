"""Location layer: span collection, canonicalization guards, deterministic
matching. The one model call is faked; everything else is pure."""
import json
from collections import Counter

import pytest

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


CACHE_LOCS = {"concepts": [
    {"name": "downtown", "kind": "named", "spans": ["downtown", "downtown area"]},
    {"name": "streets", "kind": "type", "spans": ["street"]},
]}
CACHE_KEYS = ["2:5:0", "2:5:1", "2:5:2", "2:5:3"]
CACHE_TEXTS = ["Downtown is dirty", "the street lights are out",
               "nothing here", "the downtown area needs street cleaning"]


@pytest.fixture
def loc_dir(tmp_path, monkeypatch):
    """Point the location artifacts at a tmp dir so the cache never touches
    the real ./data."""
    monkeypatch.setattr(locations, "LOCATIONS_DIR", tmp_path / "locations")
    return tmp_path / "locations"


def test_cached_sweep_matches_the_live_sweep(loc_dir):
    """The whole point of the cache: identical output, computed once. If this
    ever fails, the cache is lying about the corpus."""
    live = locations.match_locations(CACHE_LOCS, CACHE_KEYS, CACHE_TEXTS)

    first, source = locations.match_locations_cached(
        CACHE_LOCS, CACHE_KEYS, CACHE_TEXTS, "2")
    assert source == "computed"
    assert first == live

    second, source = locations.match_locations_cached(
        CACHE_LOCS, CACHE_KEYS, CACHE_TEXTS, "2")
    assert source == "cache"
    assert second == live


def test_cache_invalidates_when_a_span_changes(loc_dir):
    locations.match_locations_cached(CACHE_LOCS, CACHE_KEYS, CACHE_TEXTS, "2")
    edited = {"concepts": [
        dict(CACHE_LOCS["concepts"][0], spans=["downtown"]),   # dropped a span
        CACHE_LOCS["concepts"][1],
    ]}
    members, source = locations.match_locations_cached(
        edited, CACHE_KEYS, CACHE_TEXTS, "2")
    assert source == "computed"
    assert members == locations.match_locations(edited, CACHE_KEYS, CACHE_TEXTS)


def test_cache_invalidates_when_the_corpus_changes(loc_dir):
    locations.match_locations_cached(CACHE_LOCS, CACHE_KEYS, CACHE_TEXTS, "2")
    # same row count, one response re-ingested with different text — an mtime
    # or length check would miss this; the content fingerprint must not
    texts = list(CACHE_TEXTS)
    texts[2] = "downtown again"
    members, source = locations.match_locations_cached(
        CACHE_LOCS, CACHE_KEYS, texts, "2")
    assert source == "computed"
    assert "2:5:2" in members["downtown"]


def test_cache_survives_cosmetic_edits_to_locations_json(loc_dir):
    """Only names and spans feed the sweep, so a changed note or counter must
    not throw away eight seconds of work."""
    locations.match_locations_cached(CACHE_LOCS, CACHE_KEYS, CACHE_TEXTS, "2")
    cosmetic = {**CACHE_LOCS, "note": "reworded", "n_singleton_spans_dropped": 99}
    _, source = locations.match_locations_cached(
        cosmetic, CACHE_KEYS, CACHE_TEXTS, "2")
    assert source == "cache"


def test_corrupt_cache_is_a_miss_not_a_crash(loc_dir):
    locations.match_locations_cached(CACHE_LOCS, CACHE_KEYS, CACHE_TEXTS, "2")
    locations.members_path("2").write_text("{not json", encoding="utf-8")
    members, source = locations.match_locations_cached(
        CACHE_LOCS, CACHE_KEYS, CACHE_TEXTS, "2")
    assert source == "computed"
    assert members == locations.match_locations(CACHE_LOCS, CACHE_KEYS, CACHE_TEXTS)


def test_cache_miss_when_a_concept_is_absent_from_the_payload(loc_dir):
    """A concept with no cache entry would read as "mentioned by nobody" —
    an undercount is the one failure mode worth being paranoid about."""
    locations.match_locations_cached(CACHE_LOCS, CACHE_KEYS, CACHE_TEXTS, "2")
    path = locations.members_path("2")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["members"].pop("streets")
    path.write_text(json.dumps(payload), encoding="utf-8")
    _, source = locations.match_locations_cached(
        CACHE_LOCS, CACHE_KEYS, CACHE_TEXTS, "2")
    assert source == "computed"


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
