"""Phase 2.5: post-labeling review — evidence-backed taxonomy repair.

The labeling pass is a measurement instrument, not just output. Every way a
taxonomy can be wrong leaves a mechanical signature in the full-corpus
assignments, so the diagnostics here are pure counting: no model calls, no
state, nothing to drift. The output is a report a human reads and an edits
file a human approves. This loop is optional — the pipeline runs end to end
without it — but when an analyst does repair a taxonomy, the evidence is
gathered for them instead of hunted.

Defect -> signature -> repair cost:
  duplicate labels (co-assigned)   high overlap coefficient     zero relabeling
  duplicate labels (split)         near-identical names          zero relabeling
  missing category                 uncategorized/fit=1 cluster   relabel pool only
  orphan / mis-parented label      parent_id None                zero relabeling
  phantom label                    zero responses                zero relabeling

Merges and re-parenting never touch the model because labeling keys off child
label_ids and merges are id rewrites. Only NEW labels need a model, and only
over the missing-category pool — a few dozen responses, not the corpus.

High overlap alone is NOT proof of duplication: people genuinely mention
homelessness and street cleaning in one breath. That is why candidates go to
a human as evidence, never auto-applied.
"""
from __future__ import annotations

import json

SCHEMA_VERSION = 1
OVERLAP_THRESHOLD = 0.40   # |A∩B| / min(|A|,|B|) — report, never auto-merge
MIN_MEMBERS = 4            # ignore labels too small for overlap to mean much
POOL_TEXT_CHARS = 160

VALID_OPS = {"merge", "rename", "reparent", "add_parent", "add_label", "delete"}


# ---------------------------------------------------------------------------
# Diagnostics — pure counting
# ---------------------------------------------------------------------------


def label_members(taxonomy: dict, assignments: list[dict]) -> dict[str, set[str]]:
    members: dict[str, set[str]] = {l["label_id"]: set() for l in taxonomy["labels"]}
    for a in assignments:
        for lid in a.get("label_ids") or []:
            if lid in members:
                members[lid].add(a["response_key"])
    return members


def diagnose(taxonomy: dict, assignments: list[dict]) -> dict:
    """Deterministic defect report. Free to run, safe to re-run."""
    by_id = {l["label_id"]: l for l in taxonomy["labels"]}
    members = label_members(taxonomy, assignments)

    merge_candidates = []
    big = [lid for lid, m in members.items() if len(m) >= MIN_MEMBERS]
    for i in range(len(big)):
        for j in range(i + 1, len(big)):
            a, b = members[big[i]], members[big[j]]
            shared = len(a & b)
            if not shared:
                continue
            overlap = shared / min(len(a), len(b))
            if overlap >= OVERLAP_THRESHOLD:
                la, lb = by_id[big[i]], by_id[big[j]]
                merge_candidates.append({
                    "label_a": big[i], "name_a": la["name"], "count_a": len(a),
                    "label_b": big[j], "name_b": lb["name"], "count_b": len(b),
                    "shared": shared, "overlap": round(overlap, 2),
                    "same_parent": la.get("parent_id") == lb.get("parent_id"),
                })
    merge_candidates.sort(key=lambda c: -c["overlap"])

    names: dict[str, list[str]] = {}
    for l in taxonomy["labels"]:
        names.setdefault(l["name"].strip().lower(), []).append(l["label_id"])
    duplicate_names = [
        {"name": n, "label_ids": ids} for n, ids in names.items() if len(ids) > 1
    ]

    orphan_labels = sorted(
        ({"label_id": l["label_id"], "name": l["name"],
          "count": len(members[l["label_id"]])}
         for l in taxonomy["labels"] if l.get("parent_id") is None),
        key=lambda o: -o["count"])

    zero_count_labels = [
        {"label_id": lid, "name": by_id[lid]["name"]}
        for lid, m in members.items() if not m
    ]

    # uncategorized OR fit=1: both mean "the taxonomy did not cover this".
    # A cluster here is how a missing category announces itself.
    pool = [a["response_key"] for a in assignments
            if a.get("uncategorized") or a.get("fit") == 1]

    n = len(assignments)
    n_unc = sum(1 for a in assignments if a.get("uncategorized"))
    return {
        "schema_version": SCHEMA_VERSION,
        "n_responses": n,
        "n_labels": len(taxonomy["labels"]),
        "n_uncategorized": n_unc,
        "pct_uncategorized": round(n_unc / n, 3) if n else 0.0,
        "merge_candidates": merge_candidates,
        "duplicate_names": duplicate_names,
        "orphan_labels": orphan_labels,
        "zero_count_labels": zero_count_labels,
        "missing_category_pool": pool,
    }


def render_report(taxonomy: dict, report: dict, texts: dict[str, str]) -> str:
    """Markdown for the human reviewer. Evidence first, instructions last."""
    L = [f"# Review report — question {taxonomy['question_id']}: "
         f"\"{taxonomy.get('question_text', '')}\"",
         "",
         f"{report['n_responses']} responses · {report['n_labels']} labels · "
         f"{report['n_uncategorized']} uncategorized "
         f"({report['pct_uncategorized']:.0%}) · "
         f"pool of {len(report['missing_category_pool'])} responses the "
         f"taxonomy did not cover"]

    L += ["", "## Merge candidates (overlap = shared / smaller label)", ""]
    if report["merge_candidates"]:
        L.append("overlap | shared | label A | label B | same parent")
        L.append("---|---|---|---|---")
        for c in report["merge_candidates"]:
            L.append(f"{c['overlap']:.0%} | {c['shared']} | "
                     f"{c['name_a']} ({c['count_a']}) `{c['label_a']}` | "
                     f"{c['name_b']} ({c['count_b']}) `{c['label_b']}` | "
                     f"{'yes' if c['same_parent'] else 'no'}")
        L += ["", "High overlap can be genuine co-mention, not duplication — "
                  "same-parent + similar-name pairs are the near-certain merges."]
    else:
        L.append("(none)")

    if report["duplicate_names"]:
        L += ["", "## Duplicate label names"]
        for d in report["duplicate_names"]:
            L.append(f"- `{'`, `'.join(d['label_ids'])}` are all named "
                     f"\"{d['name']}\"")

    if report["orphan_labels"]:
        L += ["", "## Orphan labels (no parent)"]
        for o in report["orphan_labels"]:
            L.append(f"- {o['count']:>4}  {o['name']}  `{o['label_id']}`")

    if report["zero_count_labels"]:
        L += ["", "## Labels with zero responses"]
        for z in report["zero_count_labels"]:
            L.append(f"- {z['name']}  `{z['label_id']}`")

    L += ["", f"## Uncovered pool ({len(report['missing_category_pool'])} "
              "responses — uncategorized or fit=1)",
          "", "A theme repeating here is a missing category.", ""]
    for key in report["missing_category_pool"]:
        t = (texts.get(key, "(text unavailable)") or "").replace("\n", " ")
        if len(t) > POOL_TEXT_CHARS:
            t = t[:POOL_TEXT_CHARS] + "…"
        L.append(f"- `{key}` {t}")

    L += ["", "## How to act on this", "",
          "Copy ops from `edits_template.json` `suggestions` into `edits`, "
          "adjust, then run `python -m scripts.review --question "
          f"{taxonomy['question_id']} --edits <file>`. Ops: merge, rename, "
          "reparent, add_parent, add_label, delete. Keys starting with `_` "
          "are ignored. Merges/reparents cost nothing; add_label triggers a "
          "relabel of the uncovered pool only."]
    return "\n".join(L) + "\n"


def suggest_edits(report: dict) -> list[dict]:
    """Ready-to-copy ops for the mechanical cases. Suggestions only — the
    human promotes them to `edits`; nothing is ever auto-applied.

    Merge candidates are *pairwise*, so a set of them can chain: A->B and
    B->C make A and C the same label, and a later A->C pair is then a
    from==into no-op that `apply_edits` rejects outright (correctly — a
    silent no-op in an approved edits file is worse than a loud failure).
    Resolving each pair through the merges already suggested keeps the list
    something the applier will accept as a whole."""
    out = []
    survivor: dict[str, str] = {}

    def _resolve(lid: str) -> str:
        seen = set()
        while lid in survivor:
            if lid in seen:            # unreachable while we only ever point
                break                  # a merged id at its survivor, but the
            seen.add(lid)              # loop must terminate regardless
            lid = survivor[lid]
        return lid

    def _merge(src: str, dst: str, evidence: str) -> None:
        src, dst = _resolve(src), _resolve(dst)
        if src == dst:                 # already the same label after chaining
            return
        survivor[src] = dst
        out.append({"op": "merge", "from": src, "into": dst,
                    "_evidence": evidence})

    for d in report["duplicate_names"]:
        keep, *rest = d["label_ids"]
        for lid in rest:
            _merge(lid, keep, f"identical name {d['name']!r}")
    for c in report["merge_candidates"]:
        if c["same_parent"] and c["overlap"] >= 0.60:
            small, big = ((c["label_a"], c["label_b"])
                          if c["count_a"] <= c["count_b"]
                          else (c["label_b"], c["label_a"]))
            _merge(small, big,
                   f"{c['overlap']:.0%} overlap, same parent "
                   f"({c['name_a']!r} / {c['name_b']!r})")
    for z in report["zero_count_labels"]:
        if _resolve(z["label_id"]) != z["label_id"]:
            continue                   # already merged away by an op above
        out.append({"op": "delete", "label_id": z["label_id"],
                    "_evidence": f"zero responses ({z['name']!r})"})
    return out


# ---------------------------------------------------------------------------
# Edits — validated, ordered, human-approved
# ---------------------------------------------------------------------------


def apply_edits(
    taxonomy: dict, assignments: list[dict], edits: list[dict]
) -> tuple[dict, list[dict], list[str]]:
    """Apply ops in order. Returns (new_taxonomy, new_assignments, change_log).
    Raises ValueError on any invalid op — a review edit that silently no-ops
    is worse than one that fails loudly. Inputs are not mutated."""
    tax = json.loads(json.dumps(taxonomy))
    asg = json.loads(json.dumps(assignments))
    labels = {l["label_id"]: l for l in tax["labels"]}
    parents = {p["parent_id"]: p for p in tax["parents"]}
    id_map: dict[str, str] = {}   # merged/deleted id -> surviving id or ""
    log: list[str] = []

    def _resolve(lid: str) -> str:
        seen = set()
        while lid in id_map:
            if lid in seen:
                raise ValueError(f"merge cycle at {lid!r}")
            seen.add(lid)
            lid = id_map[lid]
        return lid

    def _detach(l: dict) -> None:
        pid = l.get("parent_id")
        if pid and pid in parents:
            kids = parents[pid]["child_label_ids"]
            if l["label_id"] in kids:
                kids.remove(l["label_id"])

    for e in edits:
        op = e.get("op")
        if op not in VALID_OPS:
            raise ValueError(f"unknown op {op!r} in {e}")

        if op == "merge":
            src, dst = _resolve(str(e["from"])), _resolve(str(e["into"]))
            if src == dst:
                raise ValueError(f"merge from==into after resolution: {e}")
            if src not in labels or dst not in labels:
                raise ValueError(f"merge references unknown label: {e}")
            merged = labels.pop(src)
            _detach(merged)
            tax["labels"] = [l for l in tax["labels"] if l["label_id"] != src]
            survivor = labels[dst]
            survivor.setdefault("merged_in_review", []).append(
                {"label_id": src, "name": merged["name"]})
            id_map[src] = dst
            log.append(f"merge: {merged['name']!r} ({src}) -> "
                       f"{survivor['name']!r} ({dst})")

        elif op == "delete":
            lid = _resolve(str(e["label_id"]))
            if lid not in labels:
                raise ValueError(f"delete references unknown label: {e}")
            gone = labels.pop(lid)
            _detach(gone)
            tax["labels"] = [l for l in tax["labels"] if l["label_id"] != lid]
            id_map[lid] = ""
            log.append(f"delete: {gone['name']!r} ({lid})")

        elif op == "rename":
            lid = str(e.get("label_id") or e.get("parent_id") or "")
            target = labels.get(_resolve(lid)) or parents.get(lid)
            if target is None:
                raise ValueError(f"rename references unknown id: {e}")
            log.append(f"rename: {target['name']!r} -> {e['name']!r}")
            target["name"] = str(e["name"])
            if e.get("description"):
                target["description"] = str(e["description"])

        elif op == "add_parent":
            pid = str(e["parent_id"])
            if pid in parents:
                raise ValueError(f"add_parent id already exists: {e}")
            p = {"parent_id": pid, "name": str(e["name"]),
                 "description": str(e.get("description", "")),
                 "rationale": "added in review", "child_label_ids": [],
                 "absorbed": []}
            tax["parents"].append(p)
            parents[pid] = p
            log.append(f"add_parent: {p['name']!r} ({pid})")

        elif op == "reparent":
            lid = _resolve(str(e["label_id"]))
            pid = e.get("parent_id")
            if lid not in labels:
                raise ValueError(f"reparent references unknown label: {e}")
            if pid is not None and pid not in parents:
                raise ValueError(f"reparent references unknown parent: {e}")
            l = labels[lid]
            _detach(l)
            l["parent_id"] = pid
            if pid is not None:
                parents[pid]["child_label_ids"].append(lid)
            log.append(f"reparent: {l['name']!r} ({lid}) -> "
                       f"{parents[pid]['name'] if pid else None}")

        elif op == "add_label":
            lid = str(e["label_id"])
            if lid in labels or lid in id_map:
                raise ValueError(f"add_label id already exists: {e}")
            pid = e.get("parent_id")
            if pid is not None and pid not in parents:
                raise ValueError(f"add_label references unknown parent: {e}")
            l = {"label_id": lid, "parent_id": pid, "name": str(e["name"]),
                 "description": str(e.get("description", "")),
                 "include": [], "exclude": [], "examples": [],
                 "chunk_support": None,
                 "chunk_support_note": "added in review, not from induction",
                 "singleton": False, "needs_review": False,
                 "provenance": {"source": "review_edit"}}
            tax["labels"].append(l)
            labels[lid] = l
            if pid is not None:
                parents[pid]["child_label_ids"].append(lid)
            log.append(f"add_label: {l['name']!r} ({lid})")

    # rewrite assignments through the id map; dedupe; empty -> uncategorized
    for a in asg:
        ids, seen = [], set()
        for lid in a.get("label_ids") or []:
            r = _resolve(lid) if (lid in id_map or lid in labels) else lid
            if r and r in labels and r not in seen:
                seen.add(r)
                ids.append(r)
        a["label_ids"] = ids
        if not ids:
            a["uncategorized"] = True

    tax["status"] = "reviewed"
    return tax, asg, log


def new_label_ids(edits: list[dict]) -> list[str]:
    return [str(e["label_id"]) for e in edits if e.get("op") == "add_label"]
