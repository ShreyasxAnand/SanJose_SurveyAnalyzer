"""Cost rates measured from this install's own completed runs.

The pre-run estimate used to lean on two constants written down once, in July,
against one dataset: `LABEL_USD_PER_RESPONSE = 0.10 / 500` and a flat $0.001
for the lexicon and locations stages. Every finished stage since has written a
manifest recording exactly what it spent and over how many responses. This
module reads those manifests back and turns them into the rates the estimate
uses, so the number an analyst approves is fitted to their corpus rather than
to the author's.

What is measured, and what is still modelled:

  * **labeling** — input and output tokens per *unique* response text. Unique,
    not total, because `run_labeling` collapses duplicate texts before
    batching, so a per-total-response rate silently bakes in one dataset's
    duplication rate (9.4% on the 30k production file).
  * **induction** — output tokens per chunk and model calls per chunk. Input
    is no longer estimated at all: `tokens.py` counts the real MAP prompts.
  * **lexicon / locations** — dollars per run, interpolated against corpus
    size. These manifests record no token counts, so cost is all there is to
    learn. Lexicon turns out not to scale with the corpus (a 29k-response run
    cost less than a 2k one); locations does, because its cost tracks the
    number of distinct place spans. Interpolating in log10(responses) fits
    both without pretending either is linear.

Three rules keep the sample honest, and every one of them was a real trap in
this repo's data:

  1. **Filter on `tool`, never on the directory.** `scripts.label_incremental`
     writes one manifest into BOTH the taxonomy and the labels tree, so
     globbing by directory double-counts it and credits induction with
     labeling tokens.
  2. **Skip `covers: "consolidation_only"`.** A resumed induction run's
     manifest is missing its MAP spend, so its tokens-per-response is
     meaningless rather than merely noisy.
  3. **Skip tiny runs.** The synthetic 50-row test datasets have ten unique
     texts between them and produce rates 3-8x off the real ones. They would
     otherwise dominate by count, since there are more of them than production
     runs.

A run whose latest directory is a `_review` run has no usage block at all;
those are skipped and the previous run for that question is used, so the most
recent *measurable* run per question is what counts — one per question, never
the whole history, or a question re-run seven times would outvote six others.

When there is no usable history the built-in rates below apply. They are the
pooled figures from this repo's production runs as of 2026-08-24, which is
strictly better than the single-point constants they replace, and the estimate
says which of the two it used.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .subthemes import DEFAULT_MIN_N, SUBTHEMES_DIR
from .summary import LABELS_DIR, LEXICON_DIR, LOCATIONS_DIR, TAXONOMY_DIR

# A run smaller than this is a smoke test, not evidence. 200 keeps the 17
# real labeling runs and drops the 12 toy ones; the production runs are all
# >= 1000 and dominate the pooled figure anyway.
MIN_RESPONSES = 200

# Built-in fallbacks: pooled over every production-scale run in this repo on
# 2026-08-24. Each is `sum(tokens) / sum(denominator)` across runs, not a mean
# of per-run rates — a mean would weight a 500-row run like a 9,552-row one.
BUILTIN: dict[str, Any] = {
    "source": "builtin",
    "label": {
        # 2,925,366 in / 1,706,294 out over 39,747 unique texts (29 runs)
        "input_tokens_per_unique": 73.6,
        "output_tokens_per_unique": 42.9,
        # 39,747 unique / 42,530 total responses
        "unique_ratio": 0.935,
    },
    "induce": {
        # 1,035,041 out over 364 chunks (31 runs)
        "output_tokens_per_chunk": 2843.5,
        # 627 calls over 364 chunks — MAP is one call per chunk, the rest is
        # the consolidation passes those chunks trigger
        "calls_per_chunk": 1.72,
        # run_report.candidates_proposed / rows_used, median over 31 runs.
        # The code constant this replaces is 0.30, which is high.
        "candidates_per_response": 0.26,
    },
    "subtheme": {
        # Sum of member counts for categories at or above min_n, per response,
        # over the 5 production-scale questions: 54,471 / 28,897. Memberships,
        # not responses — tagging is multi-label, so one response can be a
        # member of several eligible categories and gets sub-coded in each.
        "eligible_memberships_per_response": 1.88,
        # 54,471 memberships across 105 eligible categories. Drives how the
        # projected work splits into categories, which matters because each
        # category pays its own MAP and dedup passes.
        "members_per_eligible_category": 519.0,
        # Pooled over the 15 scripts.subthemes runs recorded on this install
        # (1,898 calls, 4,454,681 in / 2,074,696 out). Per CALL rather than per
        # response on purpose: those runs were incremental — most re-coded only
        # newly-eligible categories while their manifests recorded the whole
        # question as the denominator — so a per-response rate reads low, while
        # a call is a call.
        "input_tokens_per_call": 2347.0,
        "output_tokens_per_call": 1093.0,
    },
    # (responses, usd) from the 8 recorded runs of each stage
    "lexicon": {"points": [[18, 0.0002], [28, 0.0001], [150, 0.0022],
                           [2001, 0.0054], [28906, 0.0022]]},
    "locations": {"points": [[18, 0.0001], [28, 0.0003], [150, 0.0008],
                             [2001, 0.0059], [28906, 0.0141]]},
    # p10/p90 of each run's rate over the pooled rate, across the label runs —
    # the honest width of "about this much"
    "spread": {"low": 0.75, "high": 1.35},
}


# ---------------------------------------------------------------------------
# The active calibration — what the estimator asks
# ---------------------------------------------------------------------------


@dataclass
class Calibration:
    """Rates the estimate is built from, plus where they came from."""

    data: dict[str, Any] = field(default_factory=lambda: dict(BUILTIN))

    @property
    def source(self) -> str:
        """"measured" when this install's own runs produced these numbers,
        "builtin" when it has no usable history yet."""
        return str(self.data.get("source") or "builtin")

    @property
    def measured_utc(self) -> str:
        return str(self.data.get("measured_utc") or "")

    @property
    def sample(self) -> dict[str, int]:
        value = self.data.get("sample")
        return value if isinstance(value, dict) else {}

    def _stage(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            value = BUILTIN.get(name, {})
        return value if isinstance(value, dict) else {}

    def _number(self, stage: str, key: str) -> float:
        value = self._stage(stage).get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            value = BUILTIN[stage][key]
        return float(value)

    # -- labeling ---------------------------------------------------------

    @property
    def label_input_per_unique(self) -> float:
        return self._number("label", "input_tokens_per_unique")

    @property
    def label_output_per_unique(self) -> float:
        return self._number("label", "output_tokens_per_unique")

    @property
    def unique_ratio(self) -> float:
        """Fraction of responses that are distinct texts. Used only when the
        real texts are not to hand; when they are, they get counted."""
        ratio = self._number("label", "unique_ratio")
        return min(1.0, max(0.05, ratio))

    # -- induction --------------------------------------------------------

    @property
    def induce_output_per_chunk(self) -> float:
        return self._number("induce", "output_tokens_per_chunk")

    @property
    def induce_calls_per_chunk(self) -> float:
        return self._number("induce", "calls_per_chunk")

    @property
    def candidates_per_response(self) -> float:
        return self._number("induce", "candidates_per_response")

    # -- sub-themes (the "grandchild" pass) -------------------------------

    @property
    def eligible_memberships_per_response(self) -> float:
        return self._number("subtheme", "eligible_memberships_per_response")

    @property
    def members_per_eligible_category(self) -> float:
        return max(1.0, self._number("subtheme", "members_per_eligible_category"))

    @property
    def subtheme_input_per_call(self) -> float:
        return self._number("subtheme", "input_tokens_per_call")

    @property
    def subtheme_output_per_call(self) -> float:
        return self._number("subtheme", "output_tokens_per_call")

    # -- the flat stages --------------------------------------------------

    def flat_stage_usd(self, stage: str, n_responses: int) -> float:
        """Dollars for one lexicon or locations run over `n_responses` rows,
        interpolated between the observed runs in log10(responses).

        Log rather than linear because the observations span 18 to 28,906
        responses; on a linear axis every point below a thousand collapses onto
        the origin and the fit is decided entirely by the largest run.
        """
        points = self._stage(stage).get("points")
        if not isinstance(points, list):
            points = BUILTIN[stage]["points"]
        # Several runs over the same corpus size are the norm (a re-run after
        # a prompt tweak). Collapse them to their median rather than leaving
        # duplicate x values, which would otherwise make a zero-width segment
        # whose value depends on iteration order.
        by_size: dict[float, list[float]] = {}
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            n, usd = point
            if isinstance(n, bool) or isinstance(usd, bool):
                continue
            if not isinstance(n, (int, float)) or not isinstance(usd, (int, float)):
                continue
            if n <= 0 or usd < 0:
                continue
            by_size.setdefault(float(n), []).append(float(usd))
        clean = sorted(
            (size, _percentile(costs, 0.5)) for size, costs in by_size.items())
        if not clean:
            return 0.001
        if len(clean) == 1:
            return clean[0][1]

        x = math.log10(max(1, n_responses))
        xs = [math.log10(n) for n, _ in clean]
        ys = [usd for _, usd in clean]
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if x <= xs[i]:
                span = xs[i] - xs[i - 1]
                if span <= 0:
                    return ys[i]
                t = (x - xs[i - 1]) / span
                return ys[i - 1] + t * (ys[i] - ys[i - 1])
        return ys[-1]

    # -- the band ---------------------------------------------------------

    @property
    def spread(self) -> tuple[float, float]:
        """Multiplicative low/high bounds on a total, from the observed spread
        of per-run rates around the pooled one."""
        value = self._stage("spread")
        low = value.get("low", BUILTIN["spread"]["low"])
        high = value.get("high", BUILTIN["spread"]["high"])
        try:
            low, high = float(low), float(high)
        except (TypeError, ValueError):
            low, high = BUILTIN["spread"]["low"], BUILTIN["spread"]["high"]
        if not (0 < low <= 1 <= high):
            low, high = BUILTIN["spread"]["low"], BUILTIN["spread"]["high"]
        return low, high


def active() -> Calibration:
    """The stored calibration if this install has measured one, else built-in."""
    stored = config.calibration()
    if stored and isinstance(stored, dict):
        merged = dict(BUILTIN)
        merged.update(stored)
        merged.setdefault("source", "measured")
        return Calibration(merged)
    return Calibration(dict(BUILTIN))


# ---------------------------------------------------------------------------
# Measurement — reading the manifests back
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict[str, Any] | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _question_runs(root: Path) -> list[Path]:
    """Every `<dataset>/<question>/<run>/manifest.json` under `root`, newest
    run first within each question.

    Sorted by directory name, which is a UTC timestamp — the same ordering
    `summary.latest_run_dir` relies on.
    """
    out: list[Path] = []
    if not root.is_dir():
        return out
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for question_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            runs = sorted(
                (p for p in question_dir.iterdir() if p.is_dir()), reverse=True)
            out.extend(run / "manifest.json" for run in runs)
    return out


def _newest_per_question(root: Path, tool: str,
                         accept) -> list[dict[str, Any]]:
    """The newest qualifying manifest for each question under `root`.

    One per question, not one per run: a question re-induced seven times while
    tuning a prompt would otherwise carry seven times the weight of a question
    run once. `accept` gets the parsed manifest and returns True to take it.
    """
    seen: set[tuple[str, str]] = set()
    picked: list[dict[str, Any]] = []
    for path in _question_runs(root):
        if not path.exists():
            continue
        # <root>/<dataset>/<question>/<run>/manifest.json
        question_key = (path.parent.parent.parent.name, path.parent.parent.name)
        if question_key in seen:
            continue
        manifest = _load(path)
        if manifest is None or manifest.get("tool") != tool:
            continue
        if not (manifest.get("usage") or {}):
            continue
        if not accept(manifest):
            continue
        seen.add(question_key)
        picked.append(manifest)
    return picked


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile. Hand-rolled to keep this module free of
    a numpy import for five numbers."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (pos - low) * (ordered[high] - ordered[low])


def measure(min_responses: int = MIN_RESPONSES) -> dict[str, Any]:
    """Scan this install's manifests and return a fresh calibration payload.

    Reads only; nothing is written and no model is called. Stages with too
    little history keep their built-in rates, so a partial history produces a
    partly-measured calibration rather than nothing — and `sample` reports how
    many runs backed each figure so the number can be judged.
    """
    measured: dict[str, Any] = dict(BUILTIN)
    measured["source"] = "measured"
    measured["measured_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H-%M-%SZ")
    sample: dict[str, int] = {}

    # -- labeling ---------------------------------------------------------
    def big_enough(m: dict[str, Any]) -> bool:
        report = m.get("report") or {}
        return int(report.get("responses_total") or 0) >= min_responses

    label_runs = _newest_per_question(LABELS_DIR, "scripts.label", big_enough)
    sample["label_runs"] = len(label_runs)
    if label_runs:
        total_in = total_out = total_unique = total_responses = 0
        per_run_rates: list[float] = []
        for m in label_runs:
            usage, report = m.get("usage") or {}, m.get("report") or {}
            responses = int(report.get("responses_total") or 0)
            # v1 label reports predate the key; those runs collapsed nothing
            unique = responses - int(report.get("duplicate_responses_collapsed") or 0)
            tokens_in = int(usage.get("input_tokens") or 0)
            tokens_out = int(usage.get("output_tokens") or 0)
            if unique <= 0 or tokens_in <= 0:
                continue
            total_in += tokens_in
            total_out += tokens_out
            total_unique += unique
            total_responses += responses
            per_run_rates.append((tokens_in + tokens_out) / unique)
        if total_unique > 0:
            measured["label"] = {
                "input_tokens_per_unique": round(total_in / total_unique, 2),
                "output_tokens_per_unique": round(total_out / total_unique, 2),
                "unique_ratio": round(total_unique / max(1, total_responses), 4),
            }
            pooled = (total_in + total_out) / total_unique
            if pooled > 0 and len(per_run_rates) >= 3:
                measured["spread"] = {
                    "low": round(
                        max(0.2, _percentile(per_run_rates, 0.10) / pooled), 3),
                    "high": round(
                        min(4.0, max(1.05, _percentile(per_run_rates, 0.90) / pooled)), 3),
                }

    # -- sub-themes -------------------------------------------------------
    # Eligibility is learned from the LABELS manifests, not the sub-theme
    # ones: `report.label_counts` says exactly how many responses landed in
    # each category, which is what min_n is tested against. That also means
    # this half stays measurable on an install that has never run sub-themes.
    if label_runs:
        eligible_members = eligible_cats = covered_responses = 0
        for m in label_runs:
            report = m.get("report") or {}
            counts = report.get("label_counts") or {}
            responses = int(report.get("responses_total") or 0)
            if not isinstance(counts, dict) or responses <= 0:
                continue
            big = [int(v) for v in counts.values()
                   if isinstance(v, int) and v >= DEFAULT_MIN_N]
            eligible_members += sum(big)
            eligible_cats += len(big)
            covered_responses += responses
        if covered_responses > 0:
            block = dict(BUILTIN["subtheme"])
            block["eligible_memberships_per_response"] = round(
                eligible_members / covered_responses, 3)
            if eligible_cats > 0:
                block["members_per_eligible_category"] = round(
                    eligible_members / eligible_cats, 1)
            measured["subtheme"] = block
            sample["subtheme_questions"] = len(label_runs)

    # Per-call token rates, when this install has actually run the pass.
    sub_runs = _newest_per_question(
        SUBTHEMES_DIR, "scripts.subthemes", lambda m: True)
    sample["subtheme_runs"] = len(sub_runs)
    if sub_runs:
        total_in = total_out = total_calls = 0
        for m in sub_runs:
            usage = m.get("usage") or {}
            total_in += int(usage.get("input_tokens") or 0)
            total_out += int(usage.get("output_tokens") or 0)
            total_calls += int(usage.get("calls") or 0)
        if total_calls > 0:
            block = dict(measured.get("subtheme") or BUILTIN["subtheme"])
            block["input_tokens_per_call"] = round(total_in / total_calls, 1)
            block["output_tokens_per_call"] = round(total_out / total_calls, 1)
            measured["subtheme"] = block

    # -- induction --------------------------------------------------------
    def full_run(m: dict[str, Any]) -> bool:
        usage = m.get("usage") or {}
        # a missing `covers` predates the key and means a full run
        if usage.get("covers") == "consolidation_only":
            return False
        source = m.get("source") or {}
        return int(source.get("rows_used") or 0) >= min_responses

    induce_runs = _newest_per_question(TAXONOMY_DIR, "scripts.induce", full_run)
    sample["induce_runs"] = len(induce_runs)
    if induce_runs:
        total_out = total_chunks = total_calls = 0
        candidate_rates: list[float] = []
        for m in induce_runs:
            usage = m.get("usage") or {}
            report = m.get("run_report") or {}
            chunks = int(((report.get("chunks") or {}).get("n_chunks")) or 0)
            if chunks <= 0:
                continue
            total_out += int(usage.get("output_tokens") or 0)
            total_calls += int(usage.get("calls") or 0)
            total_chunks += chunks
            rows = int((m.get("source") or {}).get("rows_used") or 0)
            proposed = int(report.get("candidates_proposed") or 0)
            if rows > 0 and proposed > 0:
                candidate_rates.append(proposed / rows)
        if total_chunks > 0:
            block = {
                "output_tokens_per_chunk": round(total_out / total_chunks, 1),
                "calls_per_chunk": round(total_calls / total_chunks, 2),
                "candidates_per_response":
                    BUILTIN["induce"]["candidates_per_response"],
            }
            if candidate_rates:
                block["candidates_per_response"] = round(
                    _percentile(candidate_rates, 0.5), 3)
            measured["induce"] = block

    # -- the flat stages --------------------------------------------------
    for stage, root, tool in (("lexicon", LEXICON_DIR, "scripts.build_lexicon"),
                              ("locations", LOCATIONS_DIR,
                               "scripts.build_locations")):
        points: list[list[float]] = []
        if root.is_dir():
            for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                m = _load(dataset_dir / "manifest.json")
                if m is None or m.get("tool") != tool:
                    continue
                usd = (m.get("usage") or {}).get("est_cost_usd")
                responses = m.get("n_responses")
                if not isinstance(usd, (int, float)) or isinstance(usd, bool):
                    continue
                if not isinstance(responses, int) or responses <= 0:
                    continue
                points.append([float(responses), float(usd)])
        sample[f"{stage}_runs"] = len(points)
        if len(points) >= 2:
            measured[stage] = {"points": sorted(points)}

    measured["sample"] = sample
    return measured


def recalibrate() -> dict[str, Any]:
    """Measure and persist. Returns the payload that was stored."""
    payload = measure()
    config.store_calibration(payload)
    return payload





