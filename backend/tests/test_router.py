"""Phase 5 router: route validation, evidence assembly, citation resolution.
All pure functions — no model, no filesystem."""
import json

import pytest

from app import router

VALID_IDS = {"2_001", "2_002", "4_001"}
INDEX = {
    "2_001": {"label_id": "2_001", "name": "Theft", "parent_name": "safety",
              "question_id": "2"},
    "2_002": {"label_id": "2_002", "name": "Dark streets", "parent_name": None,
              "question_id": "2"},
    "4_001": {"label_id": "4_001", "name": "Downtown retail", "parent_name": "downtown",
              "question_id": "4"},
}


def test_parse_route_drops_invented_ids_and_normalizes():
    raw = json.dumps({
        "answerable": True, "route": "Retrieval", "reason": "r",
        "candidates": [
            {"label_id": "2_001", "relevance": "HIGH", "rationale": "x"},
            {"label_id": "2_001", "relevance": "high", "rationale": "dup"},
            {"label_id": "9_999", "relevance": "high", "rationale": "invented"},
            {"label_id": "2_002", "relevance": "sorta", "rationale": "y"},
        ],
        "lexicon_concepts": ["public transit", "not a concept"],
    })
    route, stats = router.parse_route_output(raw, VALID_IDS, {"public transit"})
    assert route["answerable"] and route["route"] == "retrieval"
    assert [c["label_id"] for c in route["candidates"]] == ["2_001", "2_002"]
    assert route["candidates"][0]["relevance"] == "high"
    assert route["candidates"][1]["relevance"] == "medium"   # unknown -> medium
    assert route["lexicon_concepts"] == ["public transit"]
    assert stats["invalid_label_ids"] == 1
    assert stats["invalid_concepts"] == 1


def test_parse_route_empty_selection_is_unanswerable():
    raw = json.dumps({"answerable": True, "route": "retrieval", "reason": "",
                      "candidates": [{"label_id": "bogus"}]})
    route, stats = router.parse_route_output(raw, VALID_IDS, set())
    assert route["answerable"] is False
    assert any("no valid categories" in w for w in stats["warnings"])


def test_comparative_needs_two_groups():
    raw = json.dumps({
        "answerable": True, "route": "comparative", "reason": "",
        "candidates": [{"label_id": "2_001", "relevance": "high", "rationale": ""}],
        "groups": [{"name": "only one", "label_ids": ["2_001"]}],
    })
    route, stats = router.parse_route_output(raw, VALID_IDS, set())
    assert route["route"] == "retrieval"
    assert route["groups"] == []
    assert any("downgraded" in w for w in stats["warnings"])


def _route(candidates, groups=()):
    return {"answerable": True, "route": "retrieval", "reason": "",
            "candidates": candidates, "groups": list(groups),
            "lexicon_concepts": []}


def test_gather_evidence_counts_full_but_samples_quotes():
    members = {"2_001": [f"1:2:{i}" for i in range(30)]}
    texts = {k: f"response {k}" for k in members["2_001"]}
    route = _route([{"label_id": "2_001", "relevance": "high", "rationale": ""}])
    ev = router.gather_evidence(route, INDEX, members, texts,
                                max_quotes_per_label=5, seed=1)
    assert ev["selection"][0]["count"] == 30            # count covers everything
    assert len(ev["quotes"]) == 5                       # quotes are sampled
    assert ev["sampling_notes"]["2_001"] == "showing 5 of 30"
    assert [q["n"] for q in ev["quotes"]] == [1, 2, 3, 4, 5]
    # deterministic across runs
    ev2 = router.gather_evidence(route, INDEX, members, texts,
                                 max_quotes_per_label=5, seed=1)
    assert [q["response_key"] for q in ev2["quotes"]] == \
           [q["response_key"] for q in ev["quotes"]]


def test_gather_evidence_group_counts_unique_respondents():
    members = {"2_001": ["1:2:0", "1:2:1"], "2_002": ["1:2:1", "1:2:2"]}
    texts = {k: "t" for keys in members.values() for k in keys}
    route = _route(
        [{"label_id": "2_001", "relevance": "high", "rationale": ""},
         {"label_id": "2_002", "relevance": "high", "rationale": ""}],
        groups=[{"name": "g", "label_ids": ["2_001", "2_002"]}])
    ev = router.gather_evidence(route, INDEX, members, texts)
    assert ev["group_counts"][0]["count_unique_responses"] == 3   # 1:2:1 once


def _synthetic(n_labels: int, members_per_label):
    index, members = {}, {}
    for i in range(n_labels):
        lid = f"2_{i:03d}"
        index[lid] = {"label_id": lid, "name": f"Label {i}", "parent_name": None,
                      "question_id": "2"}
        n_members = (members_per_label(i) if callable(members_per_label)
                     else members_per_label)
        members[lid] = [f"1:2:{i * 1000 + j}" for j in range(n_members)]
    texts = {k: f"response {k}" for keys in members.values() for k in keys}
    cands = [{"label_id": lid, "relevance": "high", "rationale": ""}
             for lid in index]
    return index, members, texts, cands


def test_quote_budget_caps_total_beyond_min_floor():
    # 45 candidates: the old decrement loop stopped at the floor of 3/label
    # (135 quotes, silently over the 120 budget). Now 120 // 45 = 2 per label.
    index, members, texts, cands = _synthetic(45, 30)
    ev = router.gather_evidence(_route(cands), index, members, texts, seed=1)
    assert len(ev["quotes"]) <= 120
    assert len(ev["quotes"]) == 45 * 2
    assert "_quote_budget" in ev["sampling_notes"]
    assert all(s["count"] == 30 for s in ev["selection"])  # counts untouched
    per_label = {}
    for q in ev["quotes"]:
        per_label[q["label_id"]] = per_label.get(q["label_id"], 0) + 1
    assert set(per_label.values()) == {2}


def test_quote_budget_more_candidates_than_budget():
    # 150 candidates against a 120-quote budget: only the 120 largest
    # categories get a quote (1 each); every count is still reported.
    index, members, texts, cands = _synthetic(
        150, lambda i: 2 if i < 30 else 5)
    ev = router.gather_evidence(_route(cands), index, members, texts, seed=1)
    assert len(ev["quotes"]) == 120
    quoted_lids = {q["label_id"] for q in ev["quotes"]}
    assert quoted_lids == {f"2_{i:03d}" for i in range(30, 150)}  # the largest
    assert len(ev["selection"]) == 150
    assert "the 120 largest" in ev["sampling_notes"]["_quote_budget"]


def test_quote_budget_location_mode_caps_total():
    # 60 places sharing the 120-quote budget -> 2 per place, disclosed.
    index, members, texts, cands = _synthetic(1, 600)
    lid = cands[0]["label_id"]
    location_members = {
        f"place {i:02d}": members[lid][i * 10:(i + 1) * 10] for i in range(60)
    }
    route = _route(cands)
    route["group_by"] = "location"
    ev = router.gather_evidence(route, index, members, texts, seed=1,
                                location_members=location_members,
                                location_kinds={})
    assert len(ev["quotes"]) <= 120
    assert "_quote_budget:location" in ev["sampling_notes"]


def test_resolve_citations():
    evidence = {"quotes": [
        {"n": 1, "response_key": "1:2:0", "label_id": "2_001", "text": "too many break-ins"},
        {"n": 2, "response_key": "1:2:1", "label_id": "2_001", "text": "car stolen"},
    ]}
    answer, cited, invalid = router.resolve_citations(
        "Theft dominates [1][2], and lighting [9] is bad.", evidence)
    assert [q["response_key"] for q in cited] == ["1:2:0", "1:2:1"]
    assert invalid == 1
    assert "could not be resolved" in answer
    # the body carries no Sources section — that lives in a separate block
    assert "Sources" not in answer
    sources = router.render_sources_section(cited)
    assert "`1:2:0`" in sources and "`1:2:1`" in sources
    assert router.render_sources_section([]) == ""


def test_members_by_label():
    assignments = [
        {"response_key": "a", "label_ids": ["2_001"]},
        {"response_key": "b", "label_ids": ["2_001", "2_002"]},
        {"response_key": "c", "label_ids": []},
    ]
    m = router.members_by_label(assignments)
    assert m == {"2_001": ["a", "b"], "2_002": ["b"]}


def test_parse_synth_output_requires_answer():
    assert router.parse_synth_output('{"answer_markdown": "hi"}') == "hi"
    with pytest.raises(ValueError):
        router.parse_synth_output('{"answer_markdown": ""}')


# --- location dimension: group_by + location_filter ---


def test_parse_route_location_fields():
    raw = json.dumps({
        "answerable": True, "route": "aggregate", "reason": "",
        "candidates": [{"label_id": "2_001", "relevance": "high", "rationale": ""}],
        "group_by": "Location",
        "location_filter": ["downtown", "atlantis"],
    })
    route, stats = router.parse_route_output(
        raw, VALID_IDS, set(), valid_locations={"downtown", "streets"})
    assert route["group_by"] == "location"
    assert route["location_filter"] == ["downtown"]
    assert stats["invalid_locations"] == 1


def test_parse_route_group_by_location_needs_location_layer():
    raw = json.dumps({
        "answerable": True, "route": "aggregate", "reason": "",
        "candidates": [{"label_id": "2_001", "relevance": "high", "rationale": ""}],
        "group_by": "location",
    })
    route, stats = router.parse_route_output(raw, VALID_IDS, set())
    assert route["group_by"] == "category"
    assert any("no location layer" in w for w in stats["warnings"])


def _loc_route(candidates, group_by="category", location_filter=()):
    return {**_route(candidates), "group_by": group_by,
            "location_filter": list(location_filter)}


def test_gather_evidence_location_filter_restricts_and_discloses():
    members = {"2_001": ["1:2:0", "1:2:1", "1:2:2"]}
    texts = {k: "t" for k in members["2_001"]}
    location_members = {"downtown": ["1:2:0", "1:2:2"], "streets": ["1:2:1"]}
    route = _loc_route([{"label_id": "2_001", "relevance": "high", "rationale": ""}],
                       location_filter=["downtown"])
    ev = router.gather_evidence(route, INDEX, members, texts,
                                location_members=location_members)
    assert ev["selection"][0]["count"] == 2                  # filtered
    assert ev["selection"][0]["count_unfiltered"] == 3       # disclosed
    # the in-scope union respects the filter too — coverage disclosures must
    # never quote the unfiltered corpus
    assert ev["n_unique_responses"] == 2
    assert {q["response_key"] for q in ev["quotes"]} == {"1:2:0", "1:2:2"}
    counts = router.render_counts_block(ev, [], {})
    assert "2 responses mentioning downtown" in counts
    assert "of 3 total in this category" in counts


def test_gather_evidence_group_by_location():
    members = {"2_001": ["1:2:0", "1:2:1", "1:2:2", "1:2:3"]}
    texts = {k: f"text {k}" for k in members["2_001"]}
    location_members = {"downtown": ["1:2:0", "1:2:1"], "streets": ["1:2:1"],
                        "parks": ["1:9:9"]}                  # out of scope
    kinds = {"downtown": "named", "streets": "type", "parks": "type"}
    route = _loc_route([{"label_id": "2_001", "relevance": "high", "rationale": ""}],
                       group_by="location")
    ev = router.gather_evidence(route, INDEX, members, texts,
                                location_members=location_members,
                                location_kinds=kinds)
    assert ev["location_counts"] == [
        {"name": "downtown", "kind": "named", "count": 2},
        {"name": "streets", "kind": "type", "count": 1}]
    # 2 of 4 in-scope responses named any place — the denominator
    assert ev["location_denominator"] == {"in_scope": 4, "naming_any": 2}
    # a response is quoted once, under its highest-count place, owned by a label
    assert [(q["location"], q["response_key"]) for q in ev["quotes"]] == \
        [("downtown", "1:2:0"), ("downtown", "1:2:1")]
    assert all(q["label_id"] == "2_001" for q in ev["quotes"])
    counts = router.render_counts_block(ev, [], {})
    assert "responses naming any place: 2" in counts
    assert 'place "downtown" (named): 2' in counts
    quotes = router.render_quotes_block(ev, INDEX)
    assert quotes.startswith("downtown (all 2):")
    assert "streets" not in quotes.split("downtown")[0]


def test_synth_guidance_mentions_location_dimension():
    route = _loc_route([{"label_id": "2_001", "relevance": "high", "rationale": ""}],
                       group_by="location", location_filter=["downtown"])
    ev = {"selection": [], "quotes": [], "sampling_notes": {}, "group_counts": [],
          "group_by": "location", "location_filter": ["downtown"],
          "location_counts": [], "location_denominator": None}
    system, _user = router.build_synth_prompts("q?", route, ev, INDEX, [], {})
    assert "Organize the answer by PLACE" in system
    assert "restricted to responses mentioning: downtown" in system


# --- actionability dimension: specific (concrete proposals) vs general ---


CANDIDATE_2_001 = [{"label_id": "2_001", "relevance": "high", "rationale": ""}]


def _act_route(candidates, actionability_filter="", location_filter=()):
    return {**_route(candidates), "group_by": "category",
            "location_filter": list(location_filter),
            "actionability_filter": actionability_filter}


def test_parse_route_actionability_normalizes_and_gates_on_availability():
    def parse(value, available=True):
        raw = json.dumps({
            "answerable": True, "route": "retrieval", "reason": "",
            "candidates": CANDIDATE_2_001, "actionability_filter": value})
        return router.parse_route_output(raw, VALID_IDS, set(),
                                         actionability_available=available)

    assert parse("Specific")[0]["actionability_filter"] == "specific"
    assert parse("s")[0]["actionability_filter"] == "specific"   # wire alias
    assert parse("")[0]["actionability_filter"] == ""
    assert parse("any")[0]["actionability_filter"] == ""

    route, stats = parse("mostly specific")
    assert route["actionability_filter"] == ""
    assert any("unknown actionability_filter" in w for w in stats["warnings"])

    # a filter the data can't honour is dropped, loudly — never applied to a
    # corpus where every response would fall outside it
    route, stats = parse("specific", available=False)
    assert route["actionability_filter"] == ""
    assert any("no response carries an actionability code" in w
               for w in stats["warnings"])


def test_route_prompt_only_offers_actionability_when_data_has_it():
    bare, _ = router.build_route_prompts("q?", "summary")
    assert "actionability_filter" not in bare.split("Return ONLY valid JSON")[0]
    offered, _ = router.build_route_prompts(
        "q?", "summary", actionability_counts={"specific": 12, "general": 30})
    assert '"actionability_filter"' in offered.split("Return ONLY valid JSON")[0]
    assert "42 coded responses carry the mark" in offered
    assert "30 general, 12 specific" in offered


def test_gather_evidence_actionability_filter_restricts_and_discloses():
    members = {"2_001": ["1:2:0", "1:2:1", "1:2:2", "1:2:3"]}
    texts = {k: f"text {k}" for k in members["2_001"]}
    # 1:2:3 was never coded — excluded from a filtered answer AND reported as
    # missing data rather than counted as "not specific"
    actionability = {"1:2:0": "specific", "1:2:1": "general",
                     "1:2:2": "specific"}
    ev = router.gather_evidence(_act_route(CANDIDATE_2_001, "specific"),
                                INDEX, members, texts,
                                actionability_of=actionability)
    assert ev["selection"][0]["count"] == 2              # filtered
    assert ev["selection"][0]["count_unfiltered"] == 4   # disclosed
    assert ev["n_unique_responses"] == 2
    assert {q["response_key"] for q in ev["quotes"]} == {"1:2:0", "1:2:2"}
    assert ev["actionability_denominator"] == {
        "in_scope": 4, "coded": 3, "matching": 2}

    counts = router.render_counts_block(ev, [], {})
    assert "responses in scope before the specific-only filter: 4" in counts
    assert 'responses marked "specific": 2' in counts
    assert "1 in-scope response was never marked either way" in counts
    assert "2 responses proposing a specific, concrete action" in counts


def test_actionability_composes_with_location_filter():
    """The two filters intersect, and the actionability denominator is
    measured against the location-filtered scope — not the whole corpus."""
    members = {"2_001": ["1:2:0", "1:2:1", "1:2:2", "1:2:3"]}
    texts = {k: "t" for k in members["2_001"]}
    location_members = {"downtown": ["1:2:0", "1:2:1", "1:2:3"]}
    actionability = {"1:2:0": "specific", "1:2:1": "general",
                     "1:2:2": "specific", "1:2:3": "specific"}
    ev = router.gather_evidence(
        _act_route(CANDIDATE_2_001, "specific", location_filter=["downtown"]),
        INDEX, members, texts, location_members=location_members,
        actionability_of=actionability)
    # 1:2:2 is specific but not downtown; 1:2:1 is downtown but not specific
    assert ev["selection"][0]["count"] == 2
    assert {q["response_key"] for q in ev["quotes"]} == {"1:2:0", "1:2:3"}
    # denominator is over the 3 downtown responses, not all 4
    assert ev["actionability_denominator"] == {
        "in_scope": 3, "coded": 3, "matching": 2}
    counts = router.render_counts_block(ev, [], {})
    assert ("2 responses mentioning downtown and proposing a specific, "
            "concrete action") in counts


def test_gather_evidence_without_filter_is_unchanged():
    members = {"2_001": ["1:2:0", "1:2:1"]}
    texts = {k: "t" for k in members["2_001"]}
    actionability = {"1:2:0": "specific"}
    ev = router.gather_evidence(_act_route(CANDIDATE_2_001, ""), INDEX,
                                members, texts, actionability_of=actionability)
    assert ev["selection"][0]["count"] == 2
    assert "count_unfiltered" not in ev["selection"][0]
    assert ev["actionability_denominator"] is None
    assert router.filter_phrase(ev) == ""


def test_synth_guidance_mentions_actionability_restriction():
    ev = {"selection": [], "quotes": [], "sampling_notes": {}, "group_counts": [],
          "group_by": "category", "location_filter": [],
          "actionability_filter": "specific"}
    system, _user = router.build_synth_prompts(
        "what should the city do?", _act_route(CANDIDATE_2_001, "specific"),
        ev, INDEX, [], {})
    assert "propose a concrete, implementable action" in system
    assert "covers only this subset" in system
    assert "Never present it as what all respondents said." in system


# --- event dimension: responses recounting a first-hand incident ---


def _evt_route(candidates, event_filter="", actionability_filter="",
               location_filter=()):
    return {**_route(candidates), "group_by": "category",
            "location_filter": list(location_filter),
            "actionability_filter": actionability_filter,
            "event_filter": event_filter}


def test_parse_route_event_filter_normalizes_and_gates():
    def parse(value, available=True):
        raw = json.dumps({
            "answerable": True, "route": "retrieval", "reason": "",
            "candidates": CANDIDATE_2_001, "event_filter": value})
        return router.parse_route_output(raw, VALID_IDS, set(),
                                         events_available=available)

    assert parse("reported")[0]["event_filter"] == "reported"
    assert parse(True)[0]["event_filter"] == "reported"
    assert parse("")[0]["event_filter"] == ""
    assert parse("any")[0]["event_filter"] == ""
    # the negative direction is not a supported filter — rejected, not coerced
    route, stats = parse("not_reported")
    assert route["event_filter"] == ""
    assert any("unknown event_filter" in w for w in stats["warnings"])

    route, stats = parse("reported", available=False)
    assert route["event_filter"] == ""
    assert any("no response reports a first-hand incident" in w
               for w in stats["warnings"])


def test_route_prompt_only_offers_event_filter_when_incidents_exist():
    none, _ = router.build_route_prompts("q?", "s", event_counts=(0, 500))
    assert "event_filter" not in none.split("Return ONLY valid JSON")[0]
    some, _ = router.build_route_prompts("q?", "s", event_counts=(340, 3322))
    # collapse the prompt's hard wrapping so these assertions survive a reflow
    flat = " ".join(some.split())
    assert "340 of the 3322 coded responses recount" in flat
    assert "such a question IS answerable and must not be refused" in flat
    assert "no option to select responses WITHOUT an incident" in flat


def test_gather_evidence_event_filter_excludes_uncoded_from_denominator():
    """A failed-batch row is not 'no incident reported' — it is unchecked, and
    must not silently inflate the denominator's negative case."""
    members = {"2_001": ["1:2:0", "1:2:1", "1:2:2", "1:2:3"]}
    texts = {k: f"text {k}" for k in members["2_001"]}
    events = {"1:2:0"}
    coded = {"1:2:0", "1:2:1", "1:2:2"}      # 1:2:3 was never checked
    ev = router.gather_evidence(_evt_route(CANDIDATE_2_001, "reported"),
                                INDEX, members, texts,
                                event_keys=events, event_coded_keys=coded)
    assert ev["selection"][0]["count"] == 1
    assert ev["selection"][0]["count_unfiltered"] == 4
    assert {q["response_key"] for q in ev["quotes"]} == {"1:2:0"}
    assert ev["event_denominator"] == {"in_scope": 4, "coded": 3, "matching": 1}

    counts = router.render_counts_block(ev, [], {})
    assert "recounting an incident that actually happened: 1" in counts
    assert "1 in-scope response was never checked for an incident" in counts
    assert "not the same as nothing having happened to them" in counts


def test_all_three_filters_compose_with_chained_denominators():
    """Location, then actionability, then events — each denominator measured
    against what the previous filters left."""
    members = {"2_001": [f"1:2:{i}" for i in range(6)]}
    texts = {k: "t" for k in members["2_001"]}
    location_members = {"downtown": ["1:2:0", "1:2:1", "1:2:2", "1:2:3"]}
    actionability = {"1:2:0": "specific", "1:2:1": "specific",
                     "1:2:2": "general", "1:2:3": "specific",
                     "1:2:4": "specific", "1:2:5": "specific"}
    events = {"1:2:0", "1:2:4"}              # 1:2:4 is out of the location set
    coded = set(members["2_001"])
    route = _evt_route(CANDIDATE_2_001, "reported", "specific", ["downtown"])
    ev = router.gather_evidence(route, INDEX, members, texts,
                                location_members=location_members,
                                actionability_of=actionability,
                                event_keys=events, event_coded_keys=coded)
    # downtown -> 4; of those specific -> 3; of those with an incident -> 1
    assert ev["actionability_denominator"] == {
        "in_scope": 4, "coded": 4, "matching": 3}
    assert ev["event_denominator"] == {"in_scope": 3, "coded": 3, "matching": 1}
    assert ev["selection"][0]["count"] == 1
    assert {q["response_key"] for q in ev["quotes"]} == {"1:2:0"}
    assert router.filter_phrase(ev) == (
        "mentioning downtown and proposing a specific, concrete action "
        "and recounting a first-hand incident")


def test_gather_evidence_without_event_filter_is_unchanged():
    members = {"2_001": ["1:2:0", "1:2:1"]}
    texts = {k: "t" for k in members["2_001"]}
    ev = router.gather_evidence(_evt_route(CANDIDATE_2_001, ""), INDEX,
                                members, texts, event_keys={"1:2:0"},
                                event_coded_keys={"1:2:0", "1:2:1"})
    assert ev["selection"][0]["count"] == 2
    assert ev["event_denominator"] is None
    assert "count_unfiltered" not in ev["selection"][0]


def test_synth_guidance_warns_against_inverting_the_event_filter():
    ev = {"selection": [], "quotes": [], "sampling_notes": {}, "group_counts": [],
          "group_by": "category", "location_filter": [], "event_filter": "reported"}
    system, _user = router.build_synth_prompts(
        "what have people experienced?", _evt_route(CANDIDATE_2_001, "reported"),
        ev, INDEX, [], {})
    assert "actually happened to a specific person" in system
    assert "not verified incidents" in system
    assert "Never write or imply that the other respondents had nothing" in system
