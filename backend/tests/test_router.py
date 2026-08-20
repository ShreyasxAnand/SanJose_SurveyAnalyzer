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
    assert ev["sampling_notes"]["2_001"].startswith("showing 5 of 30")
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
    # Budget passed explicitly — this pins the arithmetic, not the default.
    index, members, texts, cands = _synthetic(45, 30)
    ev = router.gather_evidence(_route(cands), index, members, texts, seed=1,
                                max_total_quotes=120)
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
    ev = router.gather_evidence(_route(cands), index, members, texts, seed=1,
                                max_total_quotes=120)
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
                                location_kinds={}, max_total_quotes=120)
    assert len(ev["quotes"]) <= 120
    assert "_quote_budget:location" in ev["sampling_notes"]


def test_composite_sample_guarantees_rare_signals():
    # 60 members, budget 10: a uniform draw misses each 1-of-60 signal ~84%
    # of the time. Coverage picks must include every distinct signal carrier:
    # the one co-labeled response, the one naming a place, the one marked
    # specific, and the one recounting an incident.
    members = {"2_001": [f"1:2:{i:02d}" for i in range(60)],
               "2_002": ["1:2:59"]}                       # rare co-label
    texts = {k: "same length text" for k in members["2_001"]}
    route = _route([{"label_id": "2_001", "relevance": "high", "rationale": ""}])
    ev = router.gather_evidence(
        route, INDEX, members, texts,
        max_quotes_per_label=10, seed=1,
        location_members={"downtown": ["1:2:58"]},        # rare place mention
        actionability_of={"1:2:57": "specific"},          # rare concrete ask
        event_keys={"1:2:56"}, event_coded_keys=set(members["2_001"]))
    quoted = {q["response_key"] for q in ev["quotes"]}
    assert {"1:2:56", "1:2:57", "1:2:58", "1:2:59"} <= quoted
    assert len(ev["quotes"]) == 10
    note = ev["sampling_notes"]["2_001"]
    assert note.startswith("showing 10 of 60") and "signal tags" in note
    # deterministic across runs
    ev2 = router.gather_evidence(
        route, INDEX, members, texts,
        max_quotes_per_label=10, seed=1,
        location_members={"downtown": ["1:2:58"]},
        actionability_of={"1:2:57": "specific"},
        event_keys={"1:2:56"}, event_coded_keys=set(members["2_001"]))
    assert [q["response_key"] for q in ev2["quotes"]] == \
           [q["response_key"] for q in ev["quotes"]]


def test_composite_sample_no_sampling_below_budget():
    keys = ["b", "a", "c"]
    picked, detail = router.composite_sample(
        keys, 5, __import__("random").Random(1), lambda k: {"len:short"})
    assert picked == ["a", "b", "c"] and detail is None


def test_composite_sample_splits_budget_and_discloses():
    # 20 keys carrying 4 distinct tags in a 6-key coverage half: coverage
    # stops once tags are exhausted and the uniform fill takes the rest.
    tags = {f"k{i:02d}": {f"tag{i % 4}"} for i in range(20)}
    picked, detail = router.composite_sample(
        sorted(tags), 12, __import__("random").Random(1), lambda k: tags[k])
    assert len(picked) == 12
    assert detail["tags_covered"] == detail["tags_total"] == 4
    assert detail["coverage_picks"] == 4                  # one per distinct tag
    assert detail["random_picks"] == 8


def test_location_filter_implicit_question_not_starved():
    # Every response to "changes to improve downtown" is about downtown by
    # construction — the downtown filter must not drop the ones that didn't
    # re-type the place name (found live: it cut q8 evidence ~87%).
    index = {
        "8_001": {"label_id": "8_001", "name": "Cleanliness", "parent_name": None,
                  "question_id": "8"},
        "6_001": {"label_id": "6_001", "name": "Crime", "parent_name": None,
                  "question_id": "6"},
    }
    members = {"8_001": ["1:8:0", "1:8:1", "1:8:2"],
               "6_001": ["1:6:0", "1:6:1"]}
    texts = {k: "t" for ks in members.values() for k in ks}
    route = _route([{"label_id": "8_001", "relevance": "high", "rationale": ""},
                    {"label_id": "6_001", "relevance": "high", "rationale": ""}])
    route["location_filter"] = ["downtown"]
    # only 1:8:0 and 1:6:0 literally mention downtown
    ev = router.gather_evidence(
        route, index, members, texts,
        location_members={"downtown": ["1:8:0", "1:6:0"]},
        location_implicit_questions={"downtown": {"8"}})
    # all of q8 survives (implicit); q6 is restricted to the literal mention
    assert ev["selection"][0]["count"] == 3        # 8_001 whole
    assert ev["selection"][1]["count"] == 1        # 6_001 filtered
    assert ev["location_filter_implicit_questions"] == ["8"]
    # without the implicit map, q8 is starved to its one literal mention
    ev2 = router.gather_evidence(
        route, index, members, texts,
        location_members={"downtown": ["1:8:0", "1:6:0"]})
    assert ev2["selection"][0]["count"] == 1
    assert ev2["location_filter_implicit_questions"] == []


def test_time_filter_denominators_and_restriction():
    members = {"2_001": [f"1:2:{i}" for i in range(10)]}
    texts = {k: "t" for k in members["2_001"]}
    route = _route([{"label_id": "2_001", "relevance": "high", "rationale": ""}])
    route["time_filter"] = "night"
    ev = router.gather_evidence(
        route, INDEX, members, texts,
        time_night_keys={"1:2:0", "1:2:1"},
        time_day_keys={"1:2:2"},
        time_mentioned_keys={"1:2:0", "1:2:1", "1:2:2"})
    assert ev["time_denominator"] == {"in_scope": 10, "mentioning": 3,
                                      "matching": 2}
    assert ev["selection"][0]["count"] == 2
    assert {q["response_key"] for q in ev["quotes"]} == {"1:2:0", "1:2:1"}
    block = router.render_counts_block(ev, [], {})
    assert "explicitly mentioning nighttime: 2" in block
    assert "never report it as the other time of day" in block


def test_normalize_time_and_availability_guard():
    assert router.normalize_time("night") == "night"
    assert router.normalize_time("Daytime") == "day"
    assert router.normalize_time("") == ""
    assert router.normalize_time("dawn patrol") == "invalid"
    raw = json.dumps({"answerable": True, "route": "retrieval", "reason": "",
                      "candidates": [{"label_id": "2_001", "relevance": "high",
                                      "rationale": ""}],
                      "time_filter": "night"})
    route, stats = router.parse_route_output(raw, VALID_IDS, set(),
                                             time_available=False)
    assert route["time_filter"] == ""
    assert any("time_filter requested" in w for w in stats["warnings"])


def test_counts_block_names_the_survey_question():
    members = {"2_001": ["1:2:0"], "4_001": ["1:4:0"]}
    texts = {"1:2:0": "a", "1:4:0": "b"}
    route = _route([{"label_id": "2_001", "relevance": "high", "rationale": ""},
                    {"label_id": "4_001", "relevance": "high", "rationale": ""}])
    ev = router.gather_evidence(route, INDEX, members, texts)
    block = router.render_counts_block(
        ev, [], {"2": 100, "4": 50},
        question_texts={"2": "what feels unsafe?", "4": "improve downtown?"})
    assert 'question 2 ("what feels unsafe?")' in block
    assert 'question 4 ("improve downtown?")' in block
    # cross-question evidence also switches on the provenance guidance
    system, _ = router.build_synth_prompts(
        "q", route, ev, INDEX, [], {"2": 100, "4": 50},
        question_texts={"2": "what feels unsafe?", "4": "improve downtown?"})
    assert "MORE THAN ONE survey question" in system


def test_resolve_citations_comma_groups():
    # flash-lite sometimes writes [1, 3] instead of [1][3]. Those citations
    # used to parse as NOTHING — absent from Sources AND not counted invalid
    # (2026-08-03 numeric-integrity audit, ds7 "Top 5 common issues?" run:
    # 6 comma-grouped citations, Sources listed 2 quotes, invalid said 0).
    evidence = {"quotes": [
        {"n": 1, "response_key": "1:2:0", "label_id": "2_001", "text": "a"},
        {"n": 2, "response_key": "1:2:1", "label_id": "2_001", "text": "b"},
        {"n": 3, "response_key": "1:2:2", "label_id": "2_001", "text": "c"},
    ]}
    answer, cited, invalid = router.resolve_citations(
        "Issues persist [1, 3], and closures lag [2, 9].", evidence)
    assert [q["n"] for q in cited] == [1, 3, 2]
    assert invalid == 1                       # the 9, flagged not dropped
    assert "1 citation(s)" in answer


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


def test_group_by_location_discloses_quotes_withheld_as_duplicates():
    """A response naming two places is quoted once, under the higher-count
    place. The second place's header must NOT then read "all n" — that tells
    the synth model it has that place's complete evidence when it does not."""
    members = {"2_001": ["1:2:0", "1:2:1", "1:2:2"]}
    texts = {"1:2:0": "downtown park", "1:2:1": "downtown", "1:2:2": "park"}
    location_members = {"downtown": ["1:2:0", "1:2:1"],
                        "park": ["1:2:0", "1:2:2"]}
    route = _loc_route([{"label_id": "2_001", "relevance": "high", "rationale": ""}],
                       group_by="location")
    ev = router.gather_evidence(route, INDEX, members, texts,
                                location_members=location_members,
                                location_kinds={"downtown": "named",
                                                "park": "type"})
    # counts still cover the full data — 1:2:0 is in both places
    assert ev["location_counts"] == [
        {"name": "downtown", "kind": "named", "count": 2},
        {"name": "park", "kind": "type", "count": 2}]
    # ...but only one of park's two responses is quoted here
    assert [(q["location"], q["response_key"]) for q in ev["quotes"]] == [
        ("downtown", "1:2:0"), ("downtown", "1:2:1"), ("park", "1:2:2")]
    assert ev["sampling_notes"]["loc:park"] == \
        "showing 1 of 2; 1 already quoted under another place"
    assert "loc:downtown" not in ev["sampling_notes"]   # downtown really is all
    quotes = router.render_quotes_block(ev, INDEX)
    assert "park (showing 1 of 2; 1 already quoted under another place):" in quotes
    assert "park (all 2):" not in quotes
    assert quotes.startswith("downtown (all 2):")


def test_category_quotes_disclose_members_with_no_stored_text():
    """A member the corpus has no text for is counted but unquotable. Without
    a note the header claimed "all 3" while showing 2."""
    members = {"2_001": ["1:2:0", "1:2:1", "1:2:9"]}
    texts = {"1:2:0": "a", "1:2:1": "b"}          # 1:2:9 dropped by a re-ingest
    route = _route([{"label_id": "2_001", "relevance": "high", "rationale": ""}])
    ev = router.gather_evidence(route, INDEX, members, texts)
    assert ev["selection"][0]["count"] == 3       # the count is unchanged
    assert ev["sampling_notes"]["2_001"] == \
        "showing 2 of 3; 1 have no stored response text"
    quotes = router.render_quotes_block(ev, INDEX)
    assert "(showing 2 of 3; 1 have no stored response text):" in quotes
    assert "(all 3)" not in quotes


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


def test_demographic_filter_or_within_field_and_across_fields():
    """The analyst's respondent filter: values within one field are OR,
    fields are AND, and `coded` counts only respondents with a value for
    EVERY filtered field — a blank cell is missing data, not a non-match."""
    members = {"2_001": [f"1:2:{i}" for i in range(8)]}
    texts = {k: "t" for k in members["2_001"]}
    route = _route([{"label_id": "2_001", "relevance": "high", "rationale": ""}])
    route["demographic_filter"] = {"Sex": ["Female"], "Age": ["18-24", "65+"]}
    demo_members = {
        "Sex": {"Female": {"1:2:0", "1:2:1", "1:2:2"},
                "Male": {"1:2:3", "1:2:4"}},
        "Age": {"18-24": {"1:2:0", "1:2:3"}, "65+": {"1:2:1"}},
    }
    demo_coded = {"Sex": {"1:2:0", "1:2:1", "1:2:2", "1:2:3", "1:2:4"},
                  "Age": {"1:2:0", "1:2:1", "1:2:3"}}
    ev = router.gather_evidence(route, INDEX, members, texts,
                                demographic_members=demo_members,
                                demographic_coded=demo_coded)
    # Female AND (18-24 OR 65+) -> {1:2:0, 1:2:1}; coded for BOTH fields -> 3
    assert ev["demographic_denominator"] == {"in_scope": 8, "coded": 3,
                                             "matching": 2}
    assert ev["selection"][0]["count"] == 2
    assert ev["selection"][0]["count_unfiltered"] == 8
    assert {q["response_key"] for q in ev["quotes"]} == {"1:2:0", "1:2:1"}
    assert router.filter_phrase(ev) == (
        "from respondents with Sex Female and "
        "from respondents with Age 18-24 or 65+")
    block = router.render_counts_block(ev, [], {})
    assert "before the demographic filter (Sex = Female; Age = 18-24 or 65+): 8" \
        in block
    assert "matching the filter: 2" in block
    assert "5 in-scope responses have no recorded value" in block
    assert "missing data, not a group to characterise" in block


def test_demographic_filter_composes_after_the_other_filters():
    """One more link in the same chain: the demographic denominator is
    measured against what the location filter left."""
    members = {"2_001": [f"1:2:{i}" for i in range(6)]}
    texts = {k: "t" for k in members["2_001"]}
    route = _route([{"label_id": "2_001", "relevance": "high", "rationale": ""}])
    route["location_filter"] = ["downtown"]
    route["demographic_filter"] = {"Sex": ["Female"]}
    ev = router.gather_evidence(
        route, INDEX, members, texts,
        location_members={"downtown": ["1:2:0", "1:2:1", "1:2:2", "1:2:3"]},
        demographic_members={"Sex": {"Female": {"1:2:0", "1:2:1", "1:2:5"}}},
        demographic_coded={"Sex": {"1:2:0", "1:2:1", "1:2:2", "1:2:5"}})
    # downtown -> 4 in scope; of those Female -> 2 (1:2:5 is out of scope)
    assert ev["demographic_denominator"] == {"in_scope": 4, "coded": 3,
                                             "matching": 2}
    assert ev["selection"][0]["count"] == 2
    assert {q["response_key"] for q in ev["quotes"]} == {"1:2:0", "1:2:1"}


def test_gather_evidence_without_demographic_filter_is_unchanged():
    members = {"2_001": ["1:2:0", "1:2:1"]}
    texts = {k: "t" for k in members["2_001"]}
    route = _route([{"label_id": "2_001", "relevance": "high", "rationale": ""}])
    ev = router.gather_evidence(
        route, INDEX, members, texts,
        demographic_members={"Sex": {"Female": {"1:2:0"}}},
        demographic_coded={"Sex": {"1:2:0", "1:2:1"}})
    assert ev["demographic_denominator"] is None
    assert ev["selection"][0]["count"] == 2
    assert "count_unfiltered" not in ev["selection"][0]


def test_route_prompt_notes_an_active_demographic_filter():
    """The router never handles the filter, but it must be told one is
    already applied — otherwise 'what do women say…' reads as unanswerable
    (observed on the first live filtered ask, 2026-08-18)."""
    _system, user = router.build_route_prompts(
        "what makes women feel unsafe?", "SUMMARY",
        demographic_filter={"Sex": ["Female"]})
    assert "already restricted the evidence to respondents with" in user
    assert "Sex = Female" in user
    assert "never mark the question unanswerable" in user
    _system2, user2 = router.build_route_prompts(
        "what makes women feel unsafe?", "SUMMARY")
    assert "already restricted" not in user2


def test_synth_guidance_states_the_demographic_restriction():
    route = _route([{"label_id": "2_001", "relevance": "high", "rationale": ""}])
    route["demographic_filter"] = {"Sex": ["Female"], "Age": ["65+"]}
    ev = {"selection": [], "quotes": [], "sampling_notes": {},
          "group_counts": [], "group_by": "category", "location_filter": [],
          "demographic_filter": route["demographic_filter"],
          "demographic_denominator": {"in_scope": 10, "coded": 8,
                                      "matching": 3}}
    system, _user = router.build_synth_prompts(
        "what do older women say?", route, ev, INDEX, [], {})
    assert ("The analyst restricted the evidence to respondents with "
            "Sex = Female; Age = 65+") in system
    assert "missing data, not a group" in system


def test_synth_guidance_warns_against_inverting_the_event_filter():
    ev = {"selection": [], "quotes": [], "sampling_notes": {}, "group_counts": [],
          "group_by": "category", "location_filter": [], "event_filter": "reported"}
    system, _user = router.build_synth_prompts(
        "what have people experienced?", _evt_route(CANDIDATE_2_001, "reported"),
        ev, INDEX, [], {})
    assert "actually happened to a specific person" in system
    assert "not verified incidents" in system
    assert "Never write or imply that the other respondents had nothing" in system


def _plan_sel(label_id, name, question_id, count,
              subs=(("aspect a", 30), ("aspect b", 20))):
    # render_section_plan returns "" unless SOME category carries sub_counts —
    # the plan only exists where the sub-theme layer gave it real denominators
    return {"label_id": label_id, "name": name, "question_id": question_id,
            "count": count, "parent_name": None,
            "sub_counts": [{"sub_label_id": f"{label_id}s{i:02d}", "name": n,
                            "count": c}
                           for i, (n, c) in enumerate(subs, 1)]}


def test_section_plan_never_asks_the_model_to_merge_lines():
    """Same-idea fusion is owned by code in BOTH branches. A prose merge rule
    used to ride along on multi-question asks; it re-licensed the model to
    overturn fusions code had declined, and any model-side merge collapses two
    numbered plan lines into one section, which plan_structure_violations then
    flags — sending a correct answer into a repair pass that undoes it."""
    # broad branch: > SECTION_PLAN_SUBTHEME_MAX_CATS categories, two questions
    sel = [_plan_sel(f"{q}_{i:03d}", f"Topic {q}{i}", q, 100 - i)
           for q in ("2", "4") for i in range(1, 4)]
    plan = router.render_section_plan(
        {"selection": sel, "group_by": "category"})
    assert "one section per category" in plan
    assert "write ONE section for both" not in plan
    assert "overrides one-section-per-line" not in plan


def test_section_plan_fuses_cross_question_duplicates_in_code():
    """Cross-question fusion happens in CODE, at the same 0.5 bar as within a
    question — no prose rule, no relaxed threshold. Real ds2 pair (q5/q6,
    Jaccard 0.83)."""
    sel = [
        _plan_sel("5_001", "Police Staffing, Funding, Presence, and Response Times",
                  "5", 400),
        _plan_sel("6_001", "Police Staffing, Presence, and Response Times", "6", 300),
        _plan_sel("5_002", "Street Lighting", "5", 200),
        _plan_sel("5_003", "Park Maintenance", "5", 150),
        _plan_sel("6_002", "Housing Costs", "6", 120),
    ]
    plan = router.render_section_plan({"selection": sel, "group_by": "category"})
    fused = [l for l in plan.splitlines() if "one section covering" in l]
    assert len(fused) == 1, plan
    # both counts stated, with their own question attribution, never summed
    assert "400 responses" in fused[0] and "300 responses" in fused[0]
    assert "survey question 5" in fused[0] and "survey question 6" in fused[0]
    assert "700" not in plan
    # unrelated categories stay separate
    assert any("Street Lighting" in l and "one section covering" not in l
               for l in plan.splitlines())


def test_section_plan_does_not_fuse_below_the_bar_in_either_direction():
    """One bar, cross-question included. Jaccard 0.4 ({public, transit, acces}
    vs {public, transit, safety, funding}) must not fuse whether the two sit in
    one question or two — a relaxed cross-question bar was tried and reverted,
    because a false fusion prints an unverified sameness claim while a missed
    one only prints two honest sections."""
    a, b = "Public Transit Access", "Public Transit Safety Funding"
    for qid_b in ("2", "4"):
        sel = [
            _plan_sel("2_001", a, "2", 400),
            _plan_sel(f"{qid_b}_009", b, qid_b, 300),
            _plan_sel("2_002", "Street Lighting", "2", 200),
            _plan_sel("2_003", "Park Maintenance", "2", 150),
            _plan_sel("2_004", "Housing Costs", "2", 120),
        ]
        plan = router.render_section_plan({"selection": sel,
                                           "group_by": "category"})
        assert "one section covering" not in plan, (qid_b, plan)


def test_section_plan_excludes_the_half_similarity_plateau():
    """LIVE ds1 pair that USED to fuse: "Traffic Safety and Infrastructure" vs
    "Bicycle Infrastructure and Safety" share {infrastructure, safety} for
    Jaccard exactly 0.5 — general traffic safety and a bicycle-specific ask,
    printed as "one section covering the same idea". ds6 repeats the shape
    ("public_transit_deficiencies" vs "public_transit_graffiti").

    Measured over 60,461 real within-dataset pairs, every known-false fusion
    sits exactly on the 0.5 plateau and every known-true one scores 0.6+, so
    the bar is strictly greater than 0.5."""
    sel = [
        _plan_sel("1_001", "Traffic Safety and Infrastructure", "1", 400),
        _plan_sel("1_002", "Bicycle Infrastructure and Safety", "1", 300),
        _plan_sel("1_003", "Street Lighting", "1", 200),
        _plan_sel("1_004", "Park Maintenance", "1", 150),
        _plan_sel("1_005", "Housing Costs", "1", 120),
    ]
    plan = router.render_section_plan({"selection": sel, "group_by": "category"})
    assert "one section covering" not in plan, plan


def test_the_narrower_ask_and_its_broader_twin_fuse_and_the_judge_agrees():
    """LIVE ds1 pair surviving the bump at 0.67: "Parking Availability" vs
    "Parking Availability and Pricing". This was pinned as a suspected residual
    false fusion — one name literally contains the other, which is exactly when
    Jaccard is highest and least informative — but the live ruling run settled
    it the other way:

        same_idea -> "Parking Availability and Access"
        "Both discuss the general lack of parking and accessibility in
         specific areas."

    So the similarity path and the judge agree here, and the suspicion was
    mine, not the data's. Recorded rather than deleted: this pair is also a
    real taxonomy merge candidate for the browse-clutter cleanup, and the
    ruling store is where that decision is already written down."""
    sel = [
        _plan_sel("1_001", "Parking Availability", "1", 400),
        _plan_sel("1_002", "Parking Availability and Pricing", "1", 300),
        _plan_sel("1_003", "Street Lighting", "1", 200),
        _plan_sel("1_004", "Park Maintenance", "1", 150),
        _plan_sel("1_005", "Housing Costs", "1", 120),
    ]
    plan = router.render_section_plan({"selection": sel, "group_by": "category"})
    fused = [l for l in plan.splitlines() if "one section covering" in l]
    assert len(fused) == 1, plan
    assert "400 responses" in fused[0] and "300 responses" in fused[0]
    assert "700" not in plan          # counts stated separately, never summed


# ---------------------------------------------------------------------------
# Plan build as a pure read over the ruling store
# ---------------------------------------------------------------------------


def _ruled(pairs: dict) -> object:
    """Minimal stand-in for rulings.RulingIndex — the plan only ever asks
    may_fuse()/fused_name(), so the store's internals stay out of these tests."""
    class _Idx:
        def may_fuse(self, a, b):
            return pairs.get(frozenset((a, b))) == "same_idea"

        def fused_name(self, a, b):
            return ""
    return _Idx()


def test_rulings_fuse_a_pair_name_similarity_would_never_reach():
    """The point of the pass: "Potholes" and "Street Repair Backlog" share no
    tokens at all, so every threshold misses them. An affirmative ruling
    fuses them anyway."""
    sel = [
        _plan_sel("1_001", "Potholes", "1", 400),
        _plan_sel("4_001", "Street Repair Backlog", "4", 300),
        _plan_sel("1_002", "Street Lighting", "1", 200),
        _plan_sel("1_003", "Park Maintenance", "1", 150),
        _plan_sel("4_002", "Housing Costs", "4", 120),
    ]
    idx = _ruled({frozenset(("1_001", "4_001")): "same_idea"})
    plan = router.render_section_plan({"selection": sel, "group_by": "category"},
                                      rulings=idx)
    fused = [l for l in plan.splitlines() if "one section covering" in l]
    assert len(fused) == 1, plan
    assert "400 responses" in fused[0] and "300 responses" in fused[0]


def test_a_distinct_ruling_beats_high_name_similarity():
    """The store must be able to REMOVE fusions, not only add them — otherwise
    every false positive the metric produces is permanent."""
    sel = [
        _plan_sel("5_001", "Police Staffing, Funding, Presence, and Response Times",
                  "5", 400),
        _plan_sel("6_001", "Police Staffing, Presence, and Response Times", "6", 300),
        _plan_sel("5_002", "Street Lighting", "5", 200),
        _plan_sel("5_003", "Park Maintenance", "5", 150),
        _plan_sel("6_002", "Housing Costs", "6", 120),
    ]
    unruled = router.render_section_plan({"selection": sel, "group_by": "category"})
    assert "one section covering" in unruled          # 0.83 — similarity fuses it
    ruled = router.render_section_plan({"selection": sel, "group_by": "category"},
                                       rulings=_ruled({}))
    assert "one section covering" not in ruled, ruled


def test_unruled_pairs_do_not_fuse_when_a_store_is_present():
    """No ruling means no fusion. With a store in play, similarity stops
    deciding entirely — it only nominated."""
    sel = [
        _plan_sel("1_001", "Graffiti and Vandalism", "1", 400),
        _plan_sel("3_001", "Graffiti and Vandalism", "3", 300),
        _plan_sel("1_002", "Street Lighting", "1", 200),
        _plan_sel("1_003", "Park Maintenance", "1", 150),
        _plan_sel("3_002", "Housing Costs", "3", 120),
    ]
    plan = router.render_section_plan({"selection": sel, "group_by": "category"},
                                      rulings=_ruled({}))
    assert "one section covering" not in plan, plan


def test_complete_linkage_never_groups_an_explicitly_distinct_pair():
    """Real intransitive triple from the ds1 live ruling run:
        Parking Availability            == Parking Availability and Pricing
        Parking Availability and Pricing == Improve parking availability and cost
        Parking Availability            != Improve parking availability and cost

    Matching only a group's representative would let the third join whenever
    the second landed first, silently overriding an explicit distinct ruling.
    Requiring same_idea with EVERY member keeps 'distinct always wins'."""
    same = {frozenset(("1_048", "1_i009")): "same_idea",
            frozenset(("1_i009", "4_004")): "same_idea"}
    # 1_048 / 4_004 deliberately absent -> ruled distinct
    sel = [
        _plan_sel("1_i009", "Parking Availability and Pricing", "1", 400),
        _plan_sel("1_048", "Parking Availability", "1", 300),
        _plan_sel("4_004", "Improve parking availability and cost", "4", 250),
        _plan_sel("1_002", "Street Lighting", "1", 200),
        _plan_sel("1_003", "Park Maintenance", "1", 150),
    ]
    plan = router.render_section_plan({"selection": sel, "group_by": "category"},
                                      rulings=_ruled(same))
    fused = [l for l in plan.splitlines() if "one section covering" in l]
    assert len(fused) == 1, plan
    # the pair that fused is the ruled one; the distinct third stays its own line
    assert "covering 2" in fused[0], fused[0]
    assert any("Improve parking availability and cost" in l
               and "one section covering" not in l
               for l in plan.splitlines()), plan
