"""
AQQAI — FastAPI Pipeline (main.py)
====================================
Full pipeline:

  POST /api/v1/query
    User Query
        ↓
    AQQAI Orchestrator
        ↓
    3 Models (parallel threads)
        ↓
    Collect 3 Responses
        ↓
    Heuristic Scorer → Score each response (RCKS)
        ↓
    Fusion Engine → Combine all responses into one
        ↓
    Return: individual scores + fused response

Other endpoints:
  GET  /api/v1/responses/{request_id}  — all scored responses for a past query
  GET  /api/v1/evaluate/{request_id}   — fused response + all scores for a past query
  GET  /health                         — system health check
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from adapters import ALL_ADAPTERS, BaseModelAdapter
from scorer import HeuristicScorer, ModelResponse, ScoredResponse
from fusion import fuse_responses

# ──────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────

app = FastAPI(
    title       = "AQQAI Orchestrator API",
    description = "Multi-model AI orchestration with heuristic scoring (RCKS) + response fusion",
    version     = "3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

scorer = HeuristicScorer()

# In-memory store — replace with PostgreSQL in production
store: dict[str, dict] = {}


# ──────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:   str
    user_id: Optional[str] = "anonymous"


class QueryResponse(BaseModel):
    request_id:      str
    query:           str
    fused_response:  str          # combined response from all models
    model_responses: list[dict]   # individual scores for each model
    total_time_ms:   float


# ──────────────────────────────────────────────────────────
# ORCHESTRATOR
# ──────────────────────────────────────────────────────────

def _call_model(adapter: BaseModelAdapter, query: str) -> ModelResponse:
    """Call one model adapter. Runs inside a thread."""
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


def orchestrate(query: str) -> tuple[list[ScoredResponse], str, float]:
    """
    Full orchestration flow:
      1. Fan out to all models in parallel (ThreadPoolExecutor)
      2. Collect all responses
      3. Score all responses through HeuristicScorer
      4. Fuse all responses into one combined response
      5. Return (all_scored, fused_response, total_time_ms)
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

    # ── Step 4: Fuse all responses ────────────────────────
    fused = fuse_responses(query, scored)

    if not fused:
        raise ValueError("All model responses failed — cannot produce fused response.")

    total_ms = round((time.time() - start) * 1000, 2)
    return scored, fused, total_ms


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
      - request_id      → use this to fetch results later
      - fused_response  → one combined response from all models
      - model_responses → individual RCKS scores for each model
      - total_time_ms   → end-to-end latency
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    try:
        scored, fused, total_ms = orchestrate(body.query)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # All model scores + content
    all_responses = [s.to_dict_with_content() for s in scored]

    # Persist for later retrieval
    store[request_id] = {
        "query":           body.query,
        "scored":          all_responses,
        "fused_response":  fused,
        "total_ms":        total_ms,
    }

    return QueryResponse(
        request_id      = request_id,
        query           = body.query,
        fused_response  = fused,
        model_responses = all_responses,
        total_time_ms   = total_ms,
    )


@app.get("/api/v1/responses/{request_id}")
def get_responses(request_id: str):
    """All individual scored responses for a past query."""
    if request_id not in store:
        raise HTTPException(status_code=404, detail="Request ID not found.")
    data = store[request_id]
    return {
        "request_id":      request_id,
        "query":           data["query"],
        "model_responses": data["scored"],
    }


@app.get("/api/v1/evaluate/{request_id}")
def get_evaluation(request_id: str):
    """Fused response + full score breakdown for a past query."""
    if request_id not in store:
        raise HTTPException(status_code=404, detail="Request ID not found.")
    data = store[request_id]
    return {
        "request_id":      request_id,
        "query":           data["query"],
        "fused_response":  data["fused_response"],
        "model_responses": data["scored"],
        "total_time_ms":   data["total_ms"],
    }


@app.get("/health")
def health():
    """System health check."""
    return {
        "status":  "ok",
        "models":  [a.model_id for a in ALL_ADAPTERS],
        "scorer":  "heuristic_rcks_v2",
        "fusion":  "extractive_sentence_transformers",
        "version": "3.0.0",
    }