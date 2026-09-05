"""Planning the sub-theme pass — the "grandchild" layer under each category.

The cheapest assertion in this file is the most valuable one: a question with
fewer responses than the member floor cannot have a single eligible category,
so it must cost exactly zero. Most datasets on a developer's machine are in
that bucket, and a plan that charged them a small positive amount for this
pass would be wrong in the direction nobody checks.
"""
import pytest

from app import calibration, subthemes
from app.induction import ResponseRow


def _rows(n, prefix="r"):
    return [ResponseRow(response_key=f"7:1:{i}", text=f"{prefix} response {i}")
            for i in range(n)]


def _taxonomy(label_ids):
    return {
        "question_text": "what would help?",
        "labels": [{"label_id": lid, "name": f"category {lid}",
                    "description": "a description of the category"}
                   for lid in label_ids],
    }


def _assignments(rows, label_ids_by_index):
    return [{"response_key": r.response_key,
             "label_ids": label_ids_by_index(i)}
            for i, r in enumerate(rows)]


CAL = calibration.Calibration(dict(calibration.BUILTIN))


# --- the floor -------------------------------------------------------------


def test_a_question_below_the_floor_costs_nothing():
    plan = subthemes.plan_subthemes(_rows(149), cal=CAL)
    # 149 responses cannot put 150 into any one category
    assert plan["total_calls"] == 0
    assert plan["est_input_tokens"] == 0
    assert plan["est_output_tokens"] == 0
    assert "150" in plan["detail"]


def test_a_question_at_the_floor_is_planned():
    plan = subthemes.plan_subthemes(_rows(400), cal=CAL)
    assert plan["total_calls"] > 0
    assert plan["est_input_tokens"] > 0


def test_no_eligible_category_costs_nothing_even_with_labels():
    rows = _rows(500)
    # 10 categories of 50 members each: none reaches the floor
    counts = {f"1_{i:03d}": 50 for i in range(10)}
    plan = subthemes.plan_subthemes(
        rows, _taxonomy(list(counts)), counts, [], cal=CAL)
    assert plan["total_calls"] == 0
    assert plan["est_input_tokens"] == 0
    assert "no category reaches" in plan["detail"]


# --- projection (no labels yet) --------------------------------------------


def test_the_projection_scales_with_the_measured_membership_share():
    small = subthemes.plan_subthemes(_rows(500), cal=CAL)
    large = subthemes.plan_subthemes(_rows(5000), cal=CAL)
    assert large["n_members"] > small["n_members"]
    assert large["total_calls"] > small["total_calls"]
    # memberships, not responses: tagging is multi-label, so one response can
    # be sub-coded inside several eligible categories
    assert large["n_members"] == pytest.approx(
        5000 * CAL.eligible_memberships_per_response, rel=0.01)


def test_the_projection_is_labelled_projected():
    plan = subthemes.plan_subthemes(_rows(2000), cal=CAL)
    assert plan["input_basis"] == "projected"
    assert plan["count_calls"] == 0
    assert "measured share" in plan["detail"]


def test_the_projection_splits_into_categories_not_one_big_one():
    """Each category pays its own MAP and dedup passes, so ten categories of
    500 cost more than one of 5,000. A projection that ignored the split would
    understate every real dataset."""
    plan = subthemes.plan_subthemes(_rows(5000), cal=CAL)
    assert plan["n_eligible_categories"] > 1
    one_big = subthemes.plan_category_calls(
        plan["n_members"], plan["n_members"], subthemes.DEFAULT_BATCH_SIZE)
    assert plan["total_calls"] > one_big["total"]


# --- planning against real labels ------------------------------------------


def test_real_categories_are_read_from_the_label_counts():
    rows = _rows(1000)
    # two eligible categories, one far below the floor
    counts = {"1_001": 600, "1_002": 300, "1_003": 20}
    assignments = _assignments(
        rows, lambda i: ["1_001"] if i < 600 else
                        (["1_002"] if i < 900 else ["1_003"]))
    plan = subthemes.plan_subthemes(
        rows, _taxonomy(list(counts)), counts, assignments,
        question_text="what would help?", cal=CAL)

    assert plan["n_eligible_categories"] == 2      # 1_003 is under the floor
    assert plan["n_members"] == 900                # 600 + 300, not 920
    assert plan["est_input_tokens"] > 0
    assert "2 categories" in plan["detail"]


def test_multi_label_responses_are_counted_once_per_category():
    rows = _rows(400)
    # every response is in BOTH categories — 800 memberships from 400 rows
    counts = {"1_001": 400, "1_002": 400}
    assignments = _assignments(rows, lambda i: ["1_001", "1_002"])
    plan = subthemes.plan_subthemes(
        rows, _taxonomy(list(counts)), counts, assignments, cal=CAL)
    assert plan["n_members"] == 800
    assert plan["n_eligible_categories"] == 2


def test_a_category_with_no_listed_rows_is_still_priced():
    """Counts and assignments can disagree after a partial run. Dropping the
    category would silently understate the plan; pricing it from the count
    keeps it visible."""
    rows = _rows(300)
    counts = {"1_001": 300}
    plan = subthemes.plan_subthemes(
        rows, _taxonomy(["1_001"]), counts, assignments=[], cal=CAL)
    assert plan["n_eligible_categories"] == 1
    assert plan["total_calls"] > 0
    assert plan["est_input_tokens"] > 0


def test_the_call_model_matches_the_cli_dry_run():
    """`scripts.subthemes.estimate_category` is the validated model; this
    plan must not invent a different one, or the browser and the terminal
    would quote different call counts for the same work."""
    for members, unique in ((530, 520), (150, 150), (5000, 4500), (1, 1)):
        calls = subthemes.plan_category_calls(members, unique, 60)
        n_map = -(-members // subthemes.SUB_CHUNK_SIZE)
        candidates = n_map * 7 * 0.6
        expected_dedup = max(1, -(-int(candidates) // 40)) + (
            1 if candidates > 40 else 0)
        assert calls["map"] == max(1, n_map)
        assert calls["dedup"] == expected_dedup
        assert calls["sublabel"] == max(1, -(-unique // 60))
        assert calls["total"] == calls["map"] + calls["dedup"] + calls["sublabel"]


def test_output_tokens_come_from_the_measured_per_call_rate():
    rows = _rows(1000)
    counts = {"1_001": 600}
    assignments = _assignments(rows, lambda i: ["1_001"] if i < 600 else [])
    plan = subthemes.plan_subthemes(
        rows, _taxonomy(["1_001"]), counts, assignments, cal=CAL)
    # no output token can be counted ahead of time, so this is the one figure
    # that must stay a measured rate
    assert plan["est_output_tokens"] == round(
        plan["total_calls"] * CAL.subtheme_output_per_call)


def test_the_projection_measures_this_corpus_duplication():
    """Sub-labeling batches deduped texts, so a corpus of 400 responses with
    17 distinct ones is one batch, not thirteen. The rows are in hand at
    projection time, so this is measured rather than assumed."""
    varied = _rows(400, prefix="varied")
    repetitive = [ResponseRow(response_key=f"7:1:{i}",
                              text=f"same answer {i % 17}")
                  for i in range(400)]

    varied_plan = subthemes.plan_subthemes(varied, cal=CAL)
    repetitive_plan = subthemes.plan_subthemes(repetitive, cal=CAL)

    # identical membership projection — duplication does not change how many
    # responses land in eligible categories
    assert varied_plan["n_members"] == repetitive_plan["n_members"]
    # but far fewer calls, because the model only sees distinct texts
    assert repetitive_plan["total_calls"] < varied_plan["total_calls"]
    assert repetitive_plan["est_input_tokens"] < varied_plan["est_input_tokens"]
