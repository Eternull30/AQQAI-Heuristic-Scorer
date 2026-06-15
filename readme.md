# AQQAI — Multi-Model AI Orchestration System

**Intern:** Jeet Tanwar  
**Project:** AQQAI (Aqua AI) — Evaluation Layer  
**Stack:** Python 3.13, FastAPI, scikit-learn, sentence-transformers  
**Status:** Week 1 + Week 2 Tasks Complete  

---

## What This System Does

AQQAI takes a user query, sends it to multiple AI models simultaneously, scores every response using a heuristic evaluation system, and returns the best response to the user.

```
User Query
    ↓
AQQAI Orchestrator
    ↓
3 Models in Parallel (Gemini · Cerebras · Mistral)
    ↓
Collect All Responses
    ↓
Heuristic Scorer → Score Each Response (RCKS)
    ↓
Return: All Scores + Winner Response
```

---

## Project Structure

```
D:\Evaluation layer\
│
├── scorer.py          # Heuristic scoring engine (RCKS dimensions)
├── adapters.py        # Model adapters (Gemini, Cerebras, Mistral)
├── main.py            # FastAPI server + endpoints
├── run.py             # CLI test runner
│
├── .env               # API keys (not committed)
├── .env.example       # Key template
├── requirements.txt   # Dependencies
│
└── outputs/           # Saved JSON results (auto-created)
```

---

## Models Used

| Model | Provider | Type | Free Limit |
|---|---|---|---|
| gemini-2.5-flash | Google AI Studio | Own format | 1000 req/day |
| gpt-oss-120b | Cerebras | OpenAI-compatible | 1M tokens/day |
| mistral-small-latest | Mistral AI | OpenAI-compatible | Free tier |

---

## Setup

### 1. Clone and create virtual environment

```bash
cd "D:\Evaluation layer"
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install sentence-transformers
```

> **Note:** sentence-transformers downloads the `all-MiniLM-L6-v2` model (~90MB) on first run. After that it runs fully offline from cache.

### 3. Add API keys

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_key_here
CEREBRAS_API_KEY=your_cerebras_key_here
MISTRAL_API_KEY=your_mistral_key_here
```

| Key | Get it from |
|---|---|
| GEMINI_API_KEY | aistudio.google.com → Get API Key |
| CEREBRAS_API_KEY | cloud.cerebras.ai → API Keys |
| MISTRAL_API_KEY | console.mistral.ai → API Keys |

---

## Running the System

### Option A — CLI (no server needed)

```bash
# Single query
python run.py --query "Explain vector databases in simple terms"

# Single query + save output to JSON
python run.py --query "What is RAG?" --save

# Interactive mode
python run.py --interactive
```

### Option B — FastAPI Server

```bash
uvicorn main:app --reload
```

Server runs at `http://127.0.0.1:8000`  
Swagger docs at `http://127.0.0.1:8000/docs`  
ReDoc at `http://127.0.0.1:8000/redoc`

---

## API Endpoints

### `POST /api/v1/query`
Submit a query. Runs the full pipeline and returns all scores + winner.

**Request:**
```json
{
  "query": "Explain vector databases in simple terms",
  "user_id": "anonymous"
}
```

**Response:**
```json
{
  "request_id": "req_a3f2c1d4e5",
  "query": "Explain vector databases in simple terms",
  "responses": [
    {
      "model_id": "mistral-small",
      "relevance": 0.34,
      "coherence": 0.71,
      "completeness": 1.00,
      "consistency": 0.80,
      "weighted_score": 0.699,
      "confidence_tier": "MEDIUM",
      "latency_ms": 4467.0,
      "success": true,
      "error": null
    }
  ],
  "winner": {
    "model_id": "mistral-small",
    "relevance": 0.34,
    "coherence": 0.71,
    "completeness": 1.00,
    "consistency": 0.80,
    "weighted_score": 0.699,
    "confidence_tier": "MEDIUM",
    "latency_ms": 4467.0,
    "success": true,
    "error": null,
    "content": "A vector database stores data as numerical embeddings..."
  },
  "total_time_ms": 5137.11
}
```

---

### `GET /api/v1/responses/{request_id}`
Fetch all scored responses for a past query.

```
GET /api/v1/responses/req_a3f2c1d4e5
```

---

### `GET /api/v1/evaluate/{request_id}`
Fetch winner + full score breakdown for a past query.

```
GET /api/v1/evaluate/req_a3f2c1d4e5
```

---

### `GET /health`
System health check. Confirms server is alive and lists registered models.

```json
{
  "status": "ok",
  "models": ["gemini-2.5-flash", "cerebras-gpt-oss", "mistral-small"],
  "scorer": "heuristic_rcks_v2",
  "version": "2.0.0"
}
```

---

## Scoring System — RCKS

Every model response is scored on 4 dimensions. The weighted score decides the winner.

```
weighted_score = (R × 0.30) + (C × 0.25) + (K × 0.30) + (S × 0.15)
```

---

### R — Relevance (weight: 30%)

**Question:** Does the response actually address what was asked?

**Method:**
- Extract keywords from the query (remove stopwords)
- Compute TF-IDF cosine similarity between query and response
- Compute direct keyword overlap (what fraction of query keywords appear in response)
- Final score: `0.6 × TF-IDF cosine + 0.4 × keyword overlap`

**Why both signals:** TF-IDF catches semantic similarity. Keyword overlap ensures the exact query terms are present. Together they cover both meaning and literal coverage.

---

### C — Coherence (weight: 25%)

**Question:** Does the response flow logically from sentence to sentence?

**Method:**
- Split response into sentences
- Compute sentence-transformers similarity between each adjacent sentence pair using `all-MiniLM-L6-v2`
- Average all similarities → coherence score
- Falls back to TF-IDF automatically if sentence-transformers unavailable

**Why sentence-transformers and not TF-IDF:** TF-IDF only measures word overlap. Two sentences can be perfectly connected while sharing zero words — TF-IDF would score them near zero. Sentence-transformers captures meaning, not just words. Example: "A vector database stores embeddings" → "These representations capture semantic meaning" shares no words but has high meaning similarity.

---

### K — Completeness (weight: 30%)

**Question:** Did the response answer all parts of the query?

**Method:**
- Decompose query into sub-parts by splitting on conjunctions and question words (and, or, how, what, why, explain, describe)
- Check what fraction of sub-parts are covered in the response
- Secondary signal: response length relative to query complexity
- Final score: `0.70 × coverage + 0.30 × length_signal`

**Length signal formula:** `expected words = sub-parts × 40`. If the response is shorter than expected, the length signal is proportionally reduced.

---

### S — Consistency (weight: 15%)

**Question:** Does the response contradict itself?

**Method:** Starts at 1.0, penalties deducted for each issue found.

| Check | Penalty | What it catches |
|---|---|---|
| Contradiction pairs | −0.08 each | "fast" and "slow" appearing in same response |
| Named entity conflict | −0.15 | Same entity described with opposing attributes in different sentences |
| Uncertainty phrases | −0.08 each | "I think", "I'm not sure", "I believe" |
| Repetition loops | −0.20 | Same 4-word sequence appearing 3+ times |

---

### Confidence Tiers

From Yashveer's Week 1 research:

| Tier | Score Range | Action |
|---|---|---|
| HIGH | ≥ 0.85 | Serve response directly |
| MEDIUM | 0.60 – 0.84 | Serve, optionally flag |
| LOW | < 0.60 | Trigger fallback or human review |

---

## Architecture Decisions

### Adapter Pattern
Every model has its own adapter class implementing `BaseModelAdapter`. The orchestrator calls `adapter.send_query(query)` without knowing which model it's talking to. Adding a new model = write one class + add to `ALL_ADAPTERS`. Nothing else changes.

### Parallel Execution
`ThreadPoolExecutor` runs all model calls simultaneously. `urllib` is synchronous so threading is used instead of `asyncio`. Total pipeline time = slowest model, not sum of all models.

### Retry Logic
- **Non-retryable (400, 401, 403, 404, 422):** Fails immediately — bad key or bad request, retrying won't help
- **Retryable (429, 500, 502, 503, 504):** Exponential backoff — waits 1s, 2s, 4s before giving up

### Failure Isolation
Each model call runs in its own thread with its own try/catch. One model failing never cancels the others. Failed responses get zeroed out and excluded from winner selection. If all models fail, the system returns a clean 503 response.

---

## Yashveer's Week 1 Research — Contributions

Three items from Yashveer's evaluation research are integrated directly into the scorer:

**1. 3-Tier Confidence System**  
HIGH / MEDIUM / LOW thresholds (0.85, 0.60) for determining what to do with a response after scoring.

**2. Uncertainty Phrase List**  
Phrases like "I think", "I'm not sure", "I believe" identified as faithfulness signals — the model signalling it may be hallucinating. Each phrase deducts 0.08 from the consistency score.

**3. Named Entity Consistency Check**  
Flagging when the same named entity is described with conflicting attributes across different sentences in the same response — a reliable hallucination signal.

---

## Known Gaps — Production Readiness

| Gap | Current State | Production Fix |
|---|---|---|
| Storage | In-memory Python dict (resets on restart) | PostgreSQL |
| Logging | Prints to terminal | Loki (Task 3) |
| Metrics | None | Prometheus + Grafana (Task 3) |
| Docker | Not containerized | Dockerfile + docker-compose (pending) |
| Relevance scoring | TF-IDF (low scores on short queries) | Sentence-transformers v2 upgrade |

---

## Sample Terminal Output

```
============================================================
Query: Explain vector databases in simple terms
============================================================
Calling 3 models in parallel...

  ✅ cerebras-gpt-oss  — 1000ms
  ✅ mistral-small     — 4645ms
  ✅ gemini-2.5-flash  — 5091ms

────────────────────────────────────────────────────────────
SCORES (ranked)
────────────────────────────────────────────────────────────
  Model                  Rel   Coh   Com   Con   Score Tier
  ──────────────────── ───── ───── ───── ───── ─────── ────────
  mistral-small         0.34  0.71  1.00  0.80  0.7180 MEDIUM 🏆
  gemini-2.5-flash      0.40  0.45  0.80  1.00  0.6206 MEDIUM
  cerebras-gpt-oss      0.26  0.76  1.00  1.00  0.7184 MEDIUM

────────────────────────────────────────────────────────────
WINNER: mistral-small (score: 0.718, tier: MEDIUM)
────────────────────────────────────────────────────────────

Total pipeline time: 5137.11ms
============================================================
```

---

## Dependencies

```
fastapi
uvicorn
pydantic
python-dotenv
scikit-learn
numpy
scipy
sentence-transformers
```

---

*Built by Jeet Tanwar — AQQAI Intern*