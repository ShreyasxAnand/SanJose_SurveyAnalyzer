"""Verification guards: numbers trace to computed counts, quoted spans exist
in their cited sources — the two integrity defects the 2026-08-12 QA review
found in shipped answers (an invented 383, an invented quotation)."""
import json

from app import verify

EVIDENCE = {
    "selection": [
        {"label_id": "5_001", "count": 3708, "count_unfiltered": 3708,
         "sub_coded": 3708, "sub_generic": 160,
         "sub_counts": [{"sub_label_id": "5_001s01", "name": "Removal",
                         "count": 1048},
                        {"sub_label_id": "5_001s08", "name": "Housing",
                         "count": 872}]},
    ],
    "group_counts": [],
    "location_counts": [],
    "uncovered_categories": [{"label_id": "5_002", "name": "Housing",
                              "count": 1510}],
    "event_denominator": {"in_scope": 1350, "coded": 1350, "matching": 275},
    "scope_coverage": {"covered": 3708, "scope_total": 9552},
    "n_unique_responses": 3708,
    "quotes": [
        {"n": 7, "label_id": "5_001",
         "text": "Please clean up the homeless situation and the crime."},
        {"n": 9, "label_id": "5_001",
         "text": "too many homeless tents around the freeway, pavements and parks"},
    ],
}


def _viol(md):
    return verify.find_violations(md, EVIDENCE, [], {"5": 9552})


def test_legit_numbers_include_counts_and_derivations():
    nums = verify.legit_numbers(EVIDENCE, [], {"5": 9552})
    assert {3708, 1048, 872, 160, 1510, 1350, 275, 9552}.issubset(nums)
    assert 1075 in nums          # coded - matching, the tally remainder
    assert 5844 in nums          # scope_total - covered
    assert 383 not in nums


def test_clean_answer_has_no_violations():
    md = ("**1048 responses** raise removal, **872** housing; "
          "the rest of 3708 responses raise it generically. "
          "One resident wrote “clean up the homeless situation” [7].")
    assert _viol(md) == []


def test_invented_number_is_flagged():
    md = "**383 total responses across five smaller sub-themes** address it."
    v = _viol(md)
    assert len(v) == 1 and v[0]["kind"] == "number" and v[0]["value"] == 383


def test_small_prose_numbers_are_ignored():
    assert _viol("the top **3** of 12 sub-themes") == []


def test_invented_quote_wording_is_flagged():
    md = ("Residents describe “junkyard-like conditions around the "
          "freeway” [9].")
    v = _viol(md)
    assert len(v) == 1 and v[0]["kind"] == "quote"


def test_real_quote_with_ellipsis_passes():
    md = ("One wrote “homeless tents around the freeway… pavements "
          "and parks” [9].")
    assert _viol(md) == []


def test_unquoted_or_uncited_text_is_not_checked():
    # quotation without a citation attached is style, not a checkable claim
    assert _viol("They want a “cleaner, safer downtown” overall.") == []


def test_repair_returns_fixed_markdown():
    class FakeClient:
        model_id = "fake"

        def complete(self, system, user):
            assert "383" in user            # violations reach the repairer
            return json.dumps({"answer_markdown": "**1048 responses** raise removal."})

    fixed = verify.repair(FakeClient(), "**383 responses** raise removal.",
                          [{"kind": "number", "value": 383, "detail": "d"}],
                          "counts", "quotes")
    assert fixed == "**1048 responses** raise removal."


def test_repair_failure_returns_none():
    class BrokenClient:
        model_id = "fake"

        def complete(self, system, user):
            return "not json at all"

    assert verify.repair(BrokenClient(), "draft", [], "c", "q") is None
