"""
AQQAI — FastAPI Pipeline (main.py)
====================================
Full pipeline (Task 2):

  POST /api/v1/query
    User Query
        ↓
    AQQAI Orchestrator
        ↓
    5 Models (parallel threads)
        ↓
    Collect 5 Responses
        ↓
    Heuristic Scorer → Score each response
        ↓
    Return: responses + scores + winner

Other endpoints:
  GET  /api/v1/responses/{request_id}  — all scored responses for a past query
  GET  /api/v1/evaluate/{request_id}   — winner + all scores for a past query
  GET  /health                         — system health check

Note: Uses ThreadPoolExecutor for parallel model calls since
      urllib is synchronous. In production with httpx installed,
      replace with asyncio.gather for better performance.
"""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from adapters import ALL_ADAPTERS, BaseModelAdapter
from scorer import HeuristicScorer, ModelResponse, ScoredResponse

# ──────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────

app = FastAPI(
    title       = "AQQAI Orchestrator API",
    description = "Multi-model AI orchestration with heuristic scoring (RCKS)",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

scorer = HeuristicScorer()

# In-memory store — replace with PostgreSQL in production
# { request_id: { query, scored_responses, winner, total_ms } }
store: dict[str, dict] = {}


# ──────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:   str
    user_id: Optional[str] = "anonymous"


class QueryResponse(BaseModel):
    request_id:    str
    query:         str
    responses:     list[dict]   # all 5 scored responses (without content)
    winner:        dict          # winner with content included
    total_time_ms: float


# ──────────────────────────────────────────────────────────
# ORCHESTRATOR
# ──────────────────────────────────────────────────────────

def _call_model(adapter: BaseModelAdapter, query: str) -> ModelResponse:
    """
    Call one model adapter synchronously.
    Runs inside a thread from ThreadPoolExecutor.
    """
    try:
        return adapter.send_query(query)
    except Exception as e:
        return ModelResponse(
            model_id   = adapter.model_id,
            content    = "",
            latency_ms = 0.0,
            success    = False,
            error      = str(e),
        )


def orchestrate(query: str) -> tuple[list[ScoredResponse], ScoredResponse, float]:
    """
    Full orchestration flow:
      1. Fan out to all 5 models in parallel (ThreadPoolExecutor)
      2. Collect all responses — failures become ModelResponse(success=False)
      3. Score all responses through HeuristicScorer
      4. Pick winner (highest weighted_score among successful responses)
      5. Return (all_scored, winner, total_time_ms)

    return_exceptions equivalent: each thread catches its own exceptions
    so one model failure never cancels the others.
    """
    start = time.time()

    # ── Step 1 + 2: Parallel model calls ──────────────────
    raw_responses: list[ModelResponse] = []

    with ThreadPoolExecutor(max_workers=len(ALL_ADAPTERS)) as pool:
        futures = {
            pool.submit(_call_model, adapter, query): adapter.model_id
            for adapter in ALL_ADAPTERS
        }
        for future in as_completed(futures):
            raw_responses.append(future.result())

    # ── Step 3: Score all responses ───────────────────────
    scored = scorer.score_all(query, raw_responses)

    # ── Step 4: Pick winner ───────────────────────────────
    winner = scorer.pick_winner(scored)

    total_ms = round((time.time() - start) * 1000, 2)
    return scored, winner, total_ms


# ──────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────

@app.post("/api/v1/query", response_model=QueryResponse)
def submit_query(body: QueryRequest):
    """
    Main endpoint — runs the full pipeline.

    Request body:
      { "query": "Explain vector databases in simple terms" }

    Returns:
      - request_id    → use this to fetch results later
      - responses     → all 5 models with RCKS scores
      - winner        → best model response with content
      - total_time_ms → end-to-end latency
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    try:
        scored, winner, total_ms = orchestrate(body.query)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Scores only (no content) for all responses
    all_scores = [s.to_dict() for s in scored]

    # Winner gets content included
    winner_dict = winner.to_dict_with_content()

    # Persist for later retrieval
    store[request_id] = {
        "query":    body.query,
        "scored":   all_scores,
        "winner":   winner_dict,
        "total_ms": total_ms,
    }

    return QueryResponse(
        request_id    = request_id,
        query         = body.query,
        responses     = all_scores,
        winner        = winner_dict,
        total_time_ms = total_ms,
    )


@app.get("/api/v1/responses/{request_id}")
def get_responses(request_id: str):
    """All scored responses for a past query."""
    if request_id not in store:
        raise HTTPException(status_code=404, detail="Request ID not found.")
    data = store[request_id]
    return {
        "request_id": request_id,
        "query":      data["query"],
        "responses":  data["scored"],
    }


@app.get("/api/v1/evaluate/{request_id}")
def get_evaluation(request_id: str):
    """Winner + full score breakdown for a past query."""
    if request_id not in store:
        raise HTTPException(status_code=404, detail="Request ID not found.")
    data = store[request_id]
    return {
        "request_id":    request_id,
        "query":         data["query"],
        "winner":        data["winner"],
        "all_scores":    data["scored"],
        "total_time_ms": data["total_ms"],
    }


@app.get("/health")
def health():
    """System health check."""
    return {
        "status":  "ok",
        "models":  [a.model_id for a in ALL_ADAPTERS],
        "scorer":  "heuristic_rcks_v2",
        "version": "2.0.0",
    }