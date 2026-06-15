"""
AQQAI — CLI Test Runner (run.py)
==================================
Test the full pipeline from the terminal without starting the server.

Usage:
  # Single query
  python run.py --query "Explain vector databases in simple terms"

  # Query + save output to JSON file
  python run.py --query "What is RAG?" --save

  # Interactive mode — keep entering queries
  python run.py --interactive

Output is printed to terminal AND optionally saved to outputs/result_<id>.json
"""

import argparse
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from adapters import ALL_ADAPTERS, BaseModelAdapter
from scorer import HeuristicScorer, ModelResponse

scorer  = HeuristicScorer()
os.makedirs("outputs", exist_ok=True)


# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────

def _call_model(adapter: BaseModelAdapter, query: str) -> ModelResponse:
    try:
        return adapter.send_query(query)
    except Exception as e:
        return ModelResponse(
            model_id=adapter.model_id,
            content="",
            latency_ms=0.0,
            success=False,
            error=str(e),
        )


def run_pipeline(query: str) -> dict:
    """Run the full pipeline for a query. Returns the result dict."""
    request_id = f"req_{uuid.uuid4().hex[:10]}"
    start      = time.time()

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    print(f"Calling {len(ALL_ADAPTERS)} models in parallel...\n")

    # Parallel model calls
    raw_responses = []
    with ThreadPoolExecutor(max_workers=len(ALL_ADAPTERS)) as pool:
        futures = {
            pool.submit(_call_model, adapter, query): adapter.model_id
            for adapter in ALL_ADAPTERS
        }
        for future in as_completed(futures):
            result = future.result()
            status = "✅" if result.success else "❌"
            print(f"  {status} {result.model_id} — {result.latency_ms:.0f}ms")
            raw_responses.append(result)

    # Score all
    scored = scorer.score_all(query, raw_responses)
    winner = scorer.pick_winner(scored)
    total_ms = round((time.time() - start) * 1000, 2)

    # Build output
    ranked = sorted(scored, key=lambda s: s.weighted_score, reverse=True)

    print(f"\n{'─'*60}")
    print("SCORES (ranked)")
    print(f"{'─'*60}")
    print(f"  {'Model':<20} {'Rel':>5} {'Coh':>5} {'Com':>5} {'Con':>5} {'Score':>7} {'Tier':<8}")
    print(f"  {'─'*20} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*7} {'─'*8}")
    for s in ranked:
        marker = " 🏆" if s.model_id == winner.model_id else ""
        print(
            f"  {s.model_id:<20} "
            f"{s.relevance:>5.2f} "
            f"{s.coherence:>5.2f} "
            f"{s.completeness:>5.2f} "
            f"{s.consistency:>5.2f} "
            f"{s.weighted_score:>7.4f} "
            f"{s.confidence_tier:<8}"
            f"{marker}"
        )

    print(f"\n{'─'*60}")
    print(f"WINNER: {winner.model_id} (score: {winner.weighted_score}, tier: {winner.confidence_tier})")
    print(f"{'─'*60}")
    print(f"Response:\n{winner.content}")
    print(f"\nTotal pipeline time: {total_ms}ms")
    print(f"{'='*60}\n")

    # Build JSON output
    result = {
        "request_id":    request_id,
        "query":         query,
        "total_time_ms": total_ms,
        "winner": {
            **winner.to_dict(),
            "content": winner.content,
        },
        "responses": [
            {**s.to_dict(), "content": s.content}
            for s in ranked
        ],
    }
    return result


def save_result(result: dict):
    """Save result to outputs/ as a JSON file."""
    filename = f"outputs/result_{result['request_id']}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to {filename}")


# ──────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AQQAI Pipeline CLI")
    parser.add_argument("--query",       type=str,  help="Query to run")
    parser.add_argument("--save",        action="store_true", help="Save output to JSON file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.interactive:
        print("AQQAI Interactive Mode — type 'exit' to quit\n")
        while True:
            query = input("Enter query: ").strip()
            if query.lower() in {"exit", "quit", "q"}:
                break
            if not query:
                continue
            result = run_pipeline(query)
            if args.save:
                save_result(result)

    elif args.query:
        result = run_pipeline(args.query)
        if args.save:
            save_result(result)

    else:
        # Default demo query if no args provided
        result = run_pipeline("Explain vector databases in simple terms")
        if args.save:
            save_result(result)


if __name__ == "__main__":
    main()