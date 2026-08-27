"""Verification guards: numbers trace to computed counts, quoted spans exist
in their cited sources — the two integrity defects the 2026-08-12 QA review
found in shipped answers (an invented 383, an invented quotation).

Nothing here repairs anything: the guards report, the answer ships as the
model wrote it. The repair pass was removed 2026-08-25."""
import re

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


def test_citation_inside_a_bold_span_is_not_read_as_a_count():
    """LIVE false positive (run 2026-08-17T16-25-33Z, affordability): the model
    bolds a whole finding sentence and ends it with its citation, so the
    bold-number scan read the citation number as a stated count. Six of that
    answer's eight "could not be verified" statements were [18] [33] [36] [42]
    [68] [75] — citations, not claims."""
    md = ("**Skyrocketing rent prices and predatory landlord practices are "
          "cited as major cost burdens for tenants [18].**")
    assert _viol(md) == []
    # multi-citation brackets too, in both spellings
    assert _viol("**Rents are the biggest burden [12][19][23].**") == []
    assert _viol("**Rents are the biggest burden [12, 19].**") == []


def test_a_real_invented_count_still_flags_alongside_a_citation():
    """The citation strip must not become a way to smuggle a bad number in."""
    md = "**383 responses** raise removal [7]."
    v = _viol(md)
    assert len(v) == 1 and v[0]["kind"] == "number" and v[0]["value"] == 383


def test_section_regex_counts_every_section_not_just_the_first():
    """LIVE false positive, same run: `^### (.+)$` under re.S let the heading
    group run past its own line and swallow the rest of the document, so a
    9-section answer measured as 1 and was reported as a structure defect."""
    body = "\n\n".join(
        f"### Section {i}\n**{n} responses** cover it.\n- a bullet [7]"
        for i, n in enumerate([1048, 872, 160], start=1))
    plan = ("1. Removal — 1048 responses\n"
            "2. Housing — 872 responses\n"
            "3. Everything else — 160 responses\n")
    assert verify.plan_structure_violations(body, plan) == []


def test_structure_violation_still_fires_when_sections_are_actually_missing():
    body = "### Only One\n**1048 responses** cover it."
    plan = "1. Removal — 1048 responses\n2. Housing — 872 responses\n"
    v = verify.plan_structure_violations(body, plan)
    assert len(v) == 1 and v[0]["kind"] == "structure"
    assert "2 lines" in v[0]["detail"] and "1 sections" in v[0]["detail"]


def test_verify_logic_hash_covers_the_detection_code():
    """The hash gates the ask cache. While it covered only the repair prompts,
    fixing a guard left every stored answer serving the old, wrong verdict."""
    import re as _re
    h = verify.verify_logic_hash()
    assert len(h) == 16
    src = verify.verify_logic_hash.__doc__ or ""
    assert src  # documented why it is over-inclusive
    # the patterns a guard fix would touch are part of the blob
    for pat in (verify._SECTION_RE.pattern, verify._CITE_SPAN_RE.pattern):
        assert isinstance(pat, str) and pat
    saved = verify._SECTION_RE
    try:
        verify._SECTION_RE = _re.compile(r"^#### ([^\n]+)\n(.*?)(?=^### |\Z)",
                                         _re.M | _re.S)
        assert verify.verify_logic_hash() != h, "a regex edit must roll the hash"
    finally:
        verify._SECTION_RE = saved
    assert verify.verify_logic_hash() == h


# --- 2026-08-25 financial-complaints answer: nine false positives, no defect --
# All nine "could not be verified" statements on that run were verifier bugs.
# The evidence below is that answer's real shape, trimmed to two sub-themes.

PLACEMENT_EVIDENCE = {
    "selection": [
        {"label_id": "5_011", "count": 520,
         "sub_counts": [
             {"sub_label_id": "5_011s01",
              "name": "General Cost of Living and Inflation", "count": 218},
             {"sub_label_id": "5_011s02",
              "name": "General Taxes, Fees, and Fines", "count": 140},
         ]},
    ],
    "uncovered_categories": [{"label_id": "5_043", "name": "Crime",
                              "count": 799}],
    "scope_coverage": {"covered": 2365, "scope_total": 9552},
    "n_unique_responses": 2365,
    # [1] really is coded to s02 and [6] to s01 in the shipped run
    "quotes": [
        {"n": 1, "label_id": "5_011", "subs": ["5_011s02"],
         "text": "Lower taxes"},
        {"n": 6, "label_id": "5_011", "subs": ["5_011s01"],
         "text": "I have a full time job and I cannot afford to live"},
    ],
}

CATEGORY_SECTION = """### General Cost of Living
**Cost of living and financial burdens received 520 responses.**
- General Cost of Living and Inflation (218 responses): Residents work
  full-time yet cannot afford to live independently [6].
- General Taxes, Fees, and Fines (140 responses): Comments demand lower
  taxes [1].
"""


def _pviol(md):
    return verify.find_violations(md, PLACEMENT_EVIDENCE, [], {"5": 9552})


def test_percent_in_a_bold_span_is_not_read_as_a_count():
    """LIVE false positive: the coverage sentence bolds "(25%)" and the
    bold-number scan read a bare 25 as a stated count — though 25% is the
    computed coverage figure, which the percent check below validates."""
    md = ("**The searched categories cover 2365 of 9552 coded responses in "
          "scope (25%).**")
    assert _pviol(md) == []


def test_label_id_in_a_bold_span_is_not_read_as_a_count():
    """LIVE false positive: "(5_043)" splits on the underscore, and int("043")
    is 43 — flagged as an untraceable count. The sibling ids 5_001/5_003/5_004
    only escaped because 1, 3 and 4 fall under MIN_CHECKED_NUMBER."""
    md = ("**Categories not searched include General Crime Reduction and "
          "Safety (5_043) with 799 responses.**")
    assert _pviol(md) == []


def test_a_real_invented_count_still_flags_beside_a_label_id():
    """The id/percent strips must not become a way to smuggle a bad number."""
    v = _pviol("**Category (5_043) accounts for 383 responses, or 61%.**")
    assert [x["kind"] for x in v] == ["number", "percent"]
    assert v[0]["value"] == 383 and v[1]["value"] == "61%"


def test_category_grain_section_checks_each_sub_theme_bullet_separately():
    """LIVE false positive: with >4 categories selected, render_section_plan
    emits one section per CATEGORY holding a bullet per sub-theme. Matching the
    heading against sub-theme names pinned the whole section to one of them —
    "General Cost of Living" scored 0.667 against "General Cost of Living and
    Inflation" because its own "General" is a stop word — so every correctly
    placed quote in the other bullets was flagged. Seven of the nine were this.
    """
    assert _pviol(CATEGORY_SECTION) == []


def test_category_grain_still_flags_a_quote_under_the_wrong_bullet():
    """The bullet-level fix must not be a way to disable the guard."""
    swapped = CATEGORY_SECTION.replace("[6]", "[X]").replace("[1]", "[6]") \
                              .replace("[X]", "[1]")
    v = [x for x in _pviol(swapped) if x["kind"] == "quote_placement"]
    assert {x["value"] for x in v} == {"[1]", "[6]"}


def test_sub_theme_grain_section_still_matches_on_its_heading():
    """With <=4 categories the sections ARE sub-themes and carry no sub-theme
    bullets, so the heading remains the unit of attribution."""
    md = ("### General Taxes, Fees, and Fines\n"
          "**140 responses** demand lower taxes.\n"
          "- Residents work full-time yet cannot afford to live [6].\n")
    v = [x for x in _pviol(md) if x["kind"] == "quote_placement"]
    assert len(v) == 1 and v[0]["value"] == "[6]"


def test_prose_bullet_does_not_capture_a_sub_theme_region():
    """A sub-theme bullet is recognised by its "(N responses)" tail; a prose
    bullet that merely mentions a count in parentheses is not one."""
    md = ("### General Taxes, Fees, and Fines\n"
          "**140 responses** demand lower taxes.\n"
          "- Smaller asks range from business rent (13) to rebates (7) [6].\n")
    v = [x for x in _pviol(md) if x["kind"] == "quote_placement"]
    assert len(v) == 1, "heading attribution must survive a prose bullet"


def test_unmatched_heading_leaves_its_bullets_unchecked():
    """Deliberate narrowness, not an oversight. Bullets refine an attribution
    the heading already made; they never create one, so this guard cannot
    raise a warning the pre-bullet code did not. Widening it would rest on
    token Jaccard over two names — the metric rulings.py falsified over 60,461
    real pairs — and on the 2026-08-25 crime answer that meant flagging [100],
    cited under "Lack of Law Enforcement and Prosecution" while coded to the
    near-synonymous "Lack of Accountability and Enforcement"."""
    md = ("### Rising Lawlessness Overall\n"          # matches no sub-theme
          "**520 responses** describe it.\n"
          "- General Cost of Living and Inflation (218 responses): taxes [1].\n")
    assert [x for x in _pviol(md) if x["kind"] == "quote_placement"] == []


def test_structure_guard_runs_on_every_answer_not_only_repaired_ones():
    """The structure check used to be reachable ONLY through the repair path
    (ask_service ran it inside `if fixed:`), so an answer that stated the wrong
    counts but quoted cleanly was never checked. That is how the 2026-08-25
    trash answer shipped with five of ten sections opening on a sub-theme's
    count instead of their category's. With repair gone the guard is
    unconditional, so this pins the behaviour ask_service depends on.

    The body below is the exact failing shape: it opens on 1048, a REAL
    sub-theme count, so every number and quote guard passes — only the plan
    count 3708 is missing."""
    body = ("### Removal\n"
            "**Residents want encampments removed.**\n"
            "- Within removal, 1048 responses focus on clearing camps [7].\n")
    plan = "1. Homelessness — 3708 responses\n"
    assert _viol(body) == [], "no number or quote defect to find"
    v = verify.plan_structure_violations(body, plan)
    assert len(v) == 1 and v[0]["kind"] == "structure"
    assert "3708" in v[0]["detail"]


def test_plan_remainder_count_is_legitimate_to_state():
    """LIVE false positive (run 2026-08-25T20-41-14Z, affordability): the plan's
    "Everything else — N responses" line is union-counted over response keys at
    plan-build time and exists nowhere in `evidence`, so a model that copied it
    exactly was flagged for inventing 935. The plan is code-computed ground
    truth — the answer is ORDERED to copy it — so its numbers are legitimate."""
    from app import router

    def ks(p, n):
        return {f"1:5:{p}{i}" for i in range(n)}

    ev = {
        "selection": [{
            "label_id": "5_002", "name": "Housing", "question_id": "5",
            "count": 300, "sub_generic": 25, "sub_coded": 290,
            "sub_counts": [
                {"sub_label_id": "5_002s01", "name": "Rent Control", "count": 120},
                {"sub_label_id": "5_002s02", "name": "Housing Supply", "count": 100},
                {"sub_label_id": "5_002s03", "name": "Senior Exemptions", "count": 40},
                {"sub_label_id": "5_002s04", "name": "Vacant Property Tax", "count": 30},
            ]}],
        "_sub_keysets": {"5_002s01": ks("a", 120), "5_002s02": ks("b", 100),
                         "5_002s03": ks("c", 40), "5_002s04": ks("d", 30)},
        "_generic_keysets": {"5_002": ks("g", 25)},
        "_sel_keysets": {"5_002": ks("a", 120) | ks("b", 100) | ks("c", 40)
                         | ks("d", 30) | ks("g", 25)},
        "group_counts": [], "location_counts": [], "quotes": [],
        "scope_coverage": {"covered": 300, "scope_total": 1000},
        "n_unique_responses": 300,
    }
    plan = router.render_section_plan(ev, max_sections=2)
    n = int(re.search(r"Everything else — ([\d,]+) responses",
                      plan).group(1).replace(",", ""))
    assert n not in verify.legit_numbers(ev, [], {}), \
        "the union count is deliberately not derivable from evidence"

    body = f"### Everything else\n**{n} responses** raise smaller sub-themes.\n"
    assert verify.find_violations(body, ev, [], {}, plan) == []
    # and the guard must still bite when the number is NOT the plan's
    bad = f"### Everything else\n**{n + 7} responses** raise smaller sub-themes.\n"
    v = verify.find_violations(bad, ev, [], {}, plan)
    assert len(v) == 1 and v[0]["value"] == n + 7
