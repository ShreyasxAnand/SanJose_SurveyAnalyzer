"""Phase 5: ask an analyst question, get an answer grounded in the responses.

From backend/, inside the surveyanalyzer conda env (needs GEMINI_API_KEY):

    python -m scripts.ask "when residents mention affordability, what specific costs are they referring to?"
    python -m scripts.ask --route-only "what makes people feel unsafe downtown?"

Flow: rebuild the Phase 4 summary (free, deterministic) -> ROUTE call picks
candidate categories and a route -> code computes counts and samples
verbatims -> SYNTH call writes the answer, citing responses that code
resolves back to response_key. Two model calls total; every number in the
answer is computed, never estimated.

The heavy lifting lives in app.ask_service, shared with the API endpoints —
the CLI and the browser run the same pipeline and write the same artifacts:

    data/answers/{dataset_id}/{run_id}/answer.md + manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app import ask_service, induction, llm, router
from app.llm import GeminiClient


def print_candidates(route: dict, ctx: ask_service.AskContext) -> None:
    print(f"\nroute: {route['route']} — {route['reason']}")
    print(f"candidates ({len(route['candidates'])}):")
    for c in sorted(route["candidates"],
                    key=lambda c: ({"high": 0, "medium": 1, "low": 2}[c["relevance"]],
                                   -len(ctx.members.get(c["label_id"], [])))):
        e = ctx.index[c["label_id"]]
        n = len(ctx.members.get(c["label_id"], []))
        print(f"  [{c['relevance']:<6}] {c['label_id']} {e['name']} (n={n}) — {c['rationale']}")
    for g in route["groups"]:
        print(f"  group {g['name']!r}: {g['label_ids']}")
    if route["lexicon_concepts"]:
        print(f"  lexicon concepts: {route['lexicon_concepts']}")
    if route.get("group_by") == "location":
        print("  grouped by PLACE (where-question)")
    if route.get("location_filter"):
        print(f"  location filter: {route['location_filter']}")
    if route.get("actionability_filter"):
        print(f"  actionability filter: {route['actionability_filter']} only")
    if route.get("event_filter"):
        print("  event filter: only responses recounting a first-hand incident")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", help="the analyst question, in plain language")
    ap.add_argument("--dataset", help="dataset id (default: the only labeled dataset)")
    ap.add_argument("--parquet")
    ap.add_argument("--route-only", action="store_true",
                    help="stop after the routing decision (one model call)")
    ap.add_argument("--model", default=None,
                    help=f"routing model (default: {llm.DEFAULT_MODEL})")
    ap.add_argument("--synth-model", default=None,
                    help="answer-writing model (default: "
                         f"{llm.DEFAULT_SYNTH_MODEL}). One call per question, "
                         "so it runs on the stronger model; pass --model's "
                         "value here to use one model for both")
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--description", default=None,
                    help="default: the export manifest's dataset_description")
    ap.add_argument("--max-quotes-per-label", type=int,
                    default=router.DEFAULT_MAX_QUOTES_PER_LABEL)
    ap.add_argument("--max-total-quotes", type=int,
                    default=router.DEFAULT_MAX_TOTAL_QUOTES)
    ap.add_argument("--actionability", choices=["any", "specific", "general"],
                    default=None,
                    help="override the router: restrict evidence to responses "
                         "proposing a concrete action (specific), to broad "
                         "concerns (general), or to neither (any)")
    ap.add_argument("--events", choices=["any", "reported"], default=None,
                    help="override the router: restrict evidence to responses "
                         "recounting a first-hand incident (reported), or to "
                         "none (any). There is no 'no incident' option — not "
                         "describing one is not evidence none occurred")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--price-in", type=float, default=0.30)
    ap.add_argument("--price-out", type=float, default=2.50)
    # Default None means "look the rate up in llm.PRICES_PER_MTOK by model
    # id"; only needed when running a model that table doesn't know.
    ap.add_argument("--synth-price-in", type=float, default=None,
                    help="$/MTok input for --synth-model (default: the "
                         "published rate for that model, if known)")
    ap.add_argument("--synth-price-out", type=float, default=None)
    args = ap.parse_args()

    try:
        dataset_id = ask_service.discover_dataset_id(args.dataset)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    args.description = induction.resolve_description(
        args.description,
        Path(args.parquet) if args.parquet
        else induction.DATA_DIR / "exports" / dataset_id / "responses.parquet")
    ctx = ask_service.load_context(dataset_id, args.parquet, args.description)

    client = GeminiClient(model=args.model, max_output_tokens=args.max_output_tokens)
    synth_model = args.synth_model or llm.DEFAULT_SYNTH_MODEL
    # One client when both stages run the same model, so usage and the cost
    # printout stay a single line instead of two identically-labelled ones
    synth_client = (client if synth_model == client.model_id else
                    GeminiClient(model=synth_model,
                                 max_output_tokens=args.max_output_tokens))
    t0 = time.time()
    print(f'question: "{args.question}"')
    print(f"dataset {dataset_id}: {len(ctx.valid_ids)} categories across "
          f"{len(ctx.question_ids)} questions")
    print(f"model: {client.model_id}" if synth_client is client else
          f"models: route={client.model_id}, answer={synth_client.model_id}")

    route, route_stats = ask_service.propose(client, args.question, ctx,
                                             args.description)
    if args.actionability is not None:
        chosen = "" if args.actionability == "any" else args.actionability
        if chosen and not ctx.actionability:
            print("  NOTE: --actionability ignored — no labels run carries the "
                  "field; re-label to populate it")
            chosen = ""
        if chosen != route["actionability_filter"]:
            print(f"  --actionability overrides the router "
                  f"({route['actionability_filter'] or 'any'} -> "
                  f"{chosen or 'any'})")
        route["actionability_filter"] = chosen
    if args.events is not None:
        chosen = "" if args.events == "any" else "reported"
        if chosen and not ctx.events:
            print("  NOTE: --events ignored — no labelled response reports a "
                  "first-hand incident")
            chosen = ""
        if chosen != route["event_filter"]:
            print(f"  --events overrides the router "
                  f"({route['event_filter'] or 'any'} -> {chosen or 'any'})")
        route["event_filter"] = chosen
    print_candidates(route, ctx)
    if route_stats["invalid_label_ids"]:
        print(f"  invalid label ids dropped: {route_stats['invalid_label_ids']}")
    for w in route_stats["warnings"]:
        print(f"  route guard: {w}")

    if not route["answerable"] or args.route_only:
        if not route["answerable"]:
            print(f"\nNOT ANSWERABLE from this data: "
                  f"{route['reason'] or 'no relevant categories'}")
        else:
            print("\n--route-only: stopping before synthesis")
        empty = {"selection": [], "quotes": [], "sampling_notes": {},
                 "group_counts": []}
        note = ask_service.process_note(route, empty, ctx, 0, [])
        run_id = ask_service.make_run_id(args.question)
        out_dir = ask_service.write_artifacts(
            run_id=run_id, question=args.question, ctx=ctx, route=route,
            route_stats=route_stats, evidence=empty,
            lex_counts=[], answer_md=None, n_invalid_citations=0,
            selection=None, client=client, description=args.description,
            elapsed=time.time() - t0, process_note=note)
    else:
        result = ask_service.answer(
            client, args.question, route, ctx, description=args.description,
            max_quotes_per_label=args.max_quotes_per_label,
            max_total_quotes=args.max_total_quotes, seed=args.seed,
            route_stats=route_stats, synth_client=synth_client)
        out_dir = result["out_dir"]
        print(f"\n{result['process_note']}")
        print(f"\n{'=' * 72}\n{result['answer_markdown']}\n{'=' * 72}")
        print(f"cited {len(result['cited'])} of {len(result['evidence']['quotes'])} "
              f"quotes shown"
              + (f"; {result['invalid_citations']} unresolvable citation(s)"
                 if result["invalid_citations"] else ""))

    total = 0.0
    unpriced = False
    billed = [(client, args.price_in, args.price_out)]
    if synth_client is not client:
        billed.append((synth_client, args.synth_price_in, args.synth_price_out))
    for c, price_in, price_out in billed:
        u = c.usage
        if not u.calls:
            continue
        if price_in is not None and price_out is not None:
            cost = u.cost_usd(price_in, price_out)
        else:
            cost = llm.price_usd(c.model_id, u.input_tokens, u.output_tokens)
        # thinking tokens are billed at the output rate and already folded
        # into output_tokens — surfaced here because they are the difference
        # between a cheap model and an expensive one, and they are invisible
        # in the answer itself
        thinking = (f" ({u.thinking_tokens:,} thinking)"
                    if u.thinking_tokens else "")
        if cost is None:
            unpriced = True
            print(f"  {c.model_id}: {u.calls} call(s), {u.input_tokens:,} in / "
                  f"{u.output_tokens:,} out{thinking} = unpriced "
                  f"(no rate for this model; pass --synth-price-in/-out)")
        else:
            total += cost
            print(f"  {c.model_id}: {u.calls} call(s), {u.input_tokens:,} in / "
                  f"{u.output_tokens:,} out{thinking} = ${cost:.4f}")
    suffix = " (partial — some models unpriced)" if unpriced else ""
    print(f"${total:.4f}{suffix}, {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
