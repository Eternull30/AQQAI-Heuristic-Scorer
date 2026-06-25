"""
AQQAI — Heuristic Scorer (scorer.py)
=====================================
Implements all four evaluation dimensions:

  R — Relevance     (TF-IDF overlap, keyword extraction)
  C — Coherence     (sentence-transformers all-MiniLM-L6-v2,
                     falls back to TF-IDF if not available)
  K — Completeness  (sub-part decomposition + length signal)
  S — Consistency   (contradiction pairs + entity conflict detection
                     + uncertainty phrases from Yashveer's notes)

Yashveer's Week 1 research — adjustments made:
  ✓ Added 3-tier confidence system (HIGH/MEDIUM/LOW) from his notes
  ✓ Uncertainty phrase list expanded from his faithfulness signal research
  ✓ Entity consistency check added (his named-entity consistency point)
  ✓ Coherence now uses sentence-transformers all-MiniLM-L6-v2
    Falls back to TF-IDF automatically if library not available

Output format (matches task spec exactly):
  {
    "model_id":        "m1",
    "relevance":       0.82,
    "coherence":       0.74,
    "completeness":    0.91,
    "consistency":     0.88,
    "weighted_score":  0.84,
    "confidence_tier": "MEDIUM"
  }
"""

import re
import string
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# ──────────────────────────────────────────────────────────
# SENTENCE-TRANSFORMERS SETUP
# Loads once at import time — not on every function call
# Falls back to TF-IDF automatically if not installed
# ──────────────────────────────────────────────────────────

_ST_MODEL = None
_ST_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    _ST_AVAILABLE = True
    print("  [scorer] sentence-transformers loaded — using all-MiniLM-L6-v2 for coherence")
except Exception:
    _ST_AVAILABLE = False
    print("  [scorer] sentence-transformers not available — falling back to TF-IDF for coherence")


# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────

WEIGHTS = {
    "relevance":    0.30,
    "coherence":    0.25,
    "completeness": 0.30,
    "consistency":  0.15,
}

TIER_THRESHOLDS = {"HIGH": 0.85, "MEDIUM": 0.60}

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","to","of","in","for",
    "on","with","at","by","from","as","into","through","and",
    "or","but","if","then","that","this","it","its","i","we",
    "you","they","he","she","not","no","so","up","out","about",
    "than","also","just","more","very","what","how","why","when",
    "where","which","who","whom","there","their","them","these",
    "those","here","some","any","all","each","both","few","more",
}


# ──────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────

@dataclass
class ModelResponse:
    """Raw response from a model adapter."""
    model_id:   str
    content:    str
    latency_ms: float = 0.0
    success:    bool  = True
    error:      Optional[str] = None


@dataclass
class ScoredResponse:
    """A ModelResponse with all dimension scores attached."""
    model_id:        str
    content:         str
    relevance:       float
    coherence:       float
    completeness:    float
    consistency:     float
    weighted_score:  float
    confidence_tier: str
    latency_ms:      float
    success:         bool
    error:           Optional[str] = None

    def to_dict(self) -> dict:
        """Scores only — no content. Used for all-models summary."""
        return {
            "model_id":        self.model_id,
            "relevance":       self.relevance,
            "coherence":       self.coherence,
            "completeness":    self.completeness,
            "consistency":     self.consistency,
            "weighted_score":  self.weighted_score,
            "confidence_tier": self.confidence_tier,
            "latency_ms":      self.latency_ms,
            "success":         self.success,
            "error":           self.error,
        }

    def to_dict_with_content(self) -> dict:
        """Scores + response text. Used for winner."""
        d = self.to_dict()
        d["content"] = self.content
        return d


# ──────────────────────────────────────────────────────────
# LOW-LEVEL HELPERS
# ──────────────────────────────────────────────────────────

def _clean_tokens(text: str) -> list[str]:
    """Lowercase, strip punctuation, return meaningful tokens."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if len(w) > 1 and w not in STOPWORDS]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Filters out very short fragments."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 12]


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity — used for relevance and as coherence fallback."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    try:
        vec   = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf = vec.fit_transform([text_a, text_b])
        return float(sklearn_cosine(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        return 0.0


def _sentence_similarity(sent_a: str, sent_b: str) -> float:
    """
    Sentence similarity for coherence scoring.

    Uses sentence-transformers all-MiniLM-L6-v2 when available.
    Falls back to TF-IDF cosine automatically.

    Why sentence-transformers is better for coherence:
    - Understands meaning, not just word overlap
    - "A dog ran fast" and "The canine sprinted" → high similarity
    - TF-IDF would score these near 0 (no shared words)
    - Real coherence is about meaning flow, not word repetition
    """
    if _ST_AVAILABLE and _ST_MODEL is not None:
        try:
            embeddings = _ST_MODEL.encode([sent_a, sent_b])
            sim = float(sklearn_cosine([embeddings[0]], [embeddings[1]])[0][0])
            return max(0.0, min(1.0, sim))
        except Exception:
            pass
    # Fallback
    return _tfidf_cosine(sent_a, sent_b)


def _extract_keywords(text: str, top_n: int = 20) -> set[str]:
    """Extract top keywords by term frequency."""
    tokens = _clean_tokens(text)
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    sorted_kw = sorted(freq, key=lambda k: freq[k], reverse=True)
    return set(sorted_kw[:top_n])


def _decompose_query(query: str) -> list[str]:
    """
    Break a query into sub-parts by splitting on conjunctions and question words.

    Example:
      "What is a vector database and how does it store data?"
      → ["What is a vector database", "how does it store data"]
    """
    query = query.strip().rstrip("?.")
    parts = re.split(
        r"\b(and|or|also|as well as|what|how|why|when|where|which|explain|describe|list)\b",
        query,
        flags=re.IGNORECASE,
    )
    delimiters = {"and","or","also","as well as","what","how","why",
                  "when","where","which","explain","describe","list"}
    cleaned = [
        p.strip() for p in parts
        if p.strip() and p.strip().lower() not in delimiters and len(p.strip()) > 5
    ]
    return cleaned if cleaned else [query]


def _confidence_tier(score: float) -> str:
    """
    3-tier confidence system from Yashveer's Week 1 research.
    HIGH   (≥0.85) — serve directly
    MEDIUM (0.60-0.84) — serve, optionally with caveat
    LOW    (<0.60)  — trigger fallback / human review
    """
    if score >= TIER_THRESHOLDS["HIGH"]:
        return "HIGH"
    if score >= TIER_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


# ──────────────────────────────────────────────────────────
# DIMENSION 1 — RELEVANCE
# ──────────────────────────────────────────────────────────

def score_relevance(query: str, response: str) -> float:
    """
    R — Relevance
    How well does the response address the query?

    Method:
    - Primary:   TF-IDF cosine similarity (semantic overlap)
    - Secondary: keyword overlap (exact query terms in response)
    - Final:     0.6 * cosine + 0.4 * keyword_overlap
    """
    if not response.strip():
        return 0.0

    cosine = _tfidf_cosine(query, response)

    query_kw    = _extract_keywords(query, top_n=15)
    resp_tokens = set(_clean_tokens(response))
    overlap     = len(query_kw & resp_tokens) / max(len(query_kw), 1)

    score = round(0.6 * cosine + 0.4 * overlap, 4)
    return min(1.0, score)


# ──────────────────────────────────────────────────────────
# DIMENSION 2 — COHERENCE
# ──────────────────────────────────────────────────────────

def score_coherence(response: str) -> float:
    """
    C — Coherence
    Does the response flow logically from sentence to sentence?

    Method:
    - Split response into sentences
    - Compute sentence-transformers similarity between each adjacent pair
      (falls back to TF-IDF if sentence-transformers not available)
    - Average all similarities → coherence score

    Why sentence-transformers here:
    TF-IDF misses meaning-based coherence. Two sentences about the
    same topic but with different words score 0 in TF-IDF even if
    they flow perfectly. Sentence-transformers captures actual meaning.
    """
    sentences = _split_sentences(response)

    if len(sentences) < 2:
        return 0.75  # single sentence — neutral score

    similarities = []
    for i in range(len(sentences) - 1):
        sim = _sentence_similarity(sentences[i], sentences[i + 1])
        similarities.append(sim)

    avg_sim = float(np.mean(similarities))

    if _ST_AVAILABLE:
        # sentence-transformers scores sit naturally in 0.2-0.9 range
        # No calibration needed — scores are already meaningful
        score = round(avg_sim, 4)
    else:
        # TF-IDF sits in 0.02-0.25 range — needs calibration
        score = min(1.0, avg_sim * 4.0 + 0.45)

    return round(min(1.0, max(0.0, score)), 4)


# ──────────────────────────────────────────────────────────
# DIMENSION 3 — COMPLETENESS
# ──────────────────────────────────────────────────────────

def score_completeness(query: str, response: str) -> float:
    """
    K — Completeness
    Does the response address all parts of the query?

    Method:
    - Decompose query into sub-parts
    - Check coverage of each sub-part in response
    - Length signal as secondary check
    - Final: 0.70 * coverage + 0.30 * length_signal
    """
    if not response.strip():
        return 0.0

    sub_parts  = _decompose_query(query)
    resp_lower = response.lower()

    covered = 0
    for part in sub_parts:
        part_keywords = [
            kw for kw in _clean_tokens(part)
            if kw not in STOPWORDS
        ]
        if any(kw in resp_lower for kw in part_keywords):
            covered += 1

    coverage = covered / max(len(sub_parts), 1)

    expected_words = max(len(sub_parts) * 40, 50)
    actual_words   = len(response.split())
    length_signal  = min(1.0, actual_words / expected_words)

    score = round(0.70 * coverage + 0.30 * length_signal, 4)
    return min(1.0, score)


# ──────────────────────────────────────────────────────────
# DIMENSION 4 — CONSISTENCY
# ──────────────────────────────────────────────────────────

CONTRADICTION_PAIRS = [
    ("fast",       "slow"),
    ("simple",     "complex"),
    ("always",     "never"),
    ("increase",   "decrease"),
    ("efficient",  "inefficient"),
    ("safe",       "dangerous"),
    ("accurate",   "inaccurate"),
    ("reliable",   "unreliable"),
    ("easy",       "difficult"),
    ("cheap",      "expensive"),
    ("high",       "low"),
    ("large",      "small"),
    ("best",       "worst"),
    ("strong",     "weak"),
    ("faster",     "slower"),
    ("better",     "worse"),
]

UNCERTAINTY_PHRASES = [
    "i'm not sure",   "i am not sure",
    "i think",        "i believe",
    "might be wrong", "could be wrong",
    "i don't know",   "i do not know",
    "to the best of my knowledge",
    "i'm not certain","i am not certain",
    "i may be wrong", "i cannot be sure",
    "not 100% sure",  "not entirely sure",
]


def score_consistency(response: str) -> float:
    """
    S — Consistency
    Does the response contradict itself?

    Check 1 — Contradiction pairs (penalty: 0.08 each)
    Check 2 — Named entity with conflicting attributes (penalty: 0.15)
    Check 3 — Uncertainty phrases (penalty: 0.08 each)
    Check 4 — Repetition loops / degenerate output (penalty: 0.20)
    """
    if not response.strip():
        return 0.0

    score = 1.0
    lower = response.lower()

    # Check 1: Contradiction pairs
    for w1, w2 in CONTRADICTION_PAIRS:
        if re.search(rf"\b{w1}\b", lower) and re.search(rf"\b{w2}\b", lower):
            score -= 0.08

    # Check 2: Named entity consistency
    sentences = _split_sentences(response)
    entity_attributes: dict[str, list[str]] = {}

    for sent in sentences:
        matches = re.findall(
            r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+(?:is|are|was|were)\s+(\w+)",
            sent
        )
        for entity, attribute in matches:
            entity_lower = entity.lower()
            if entity_lower not in entity_attributes:
                entity_attributes[entity_lower] = []
            entity_attributes[entity_lower].append(attribute.lower())

    for entity, attrs in entity_attributes.items():
        attr_set = set(attrs)
        for w1, w2 in CONTRADICTION_PAIRS:
            if w1 in attr_set and w2 in attr_set:
                score -= 0.15
                break

    # Check 3: Uncertainty phrases
    for phrase in UNCERTAINTY_PHRASES:
        if phrase in lower:
            score -= 0.08

    # Check 4: Repetition loops
    words = lower.split()
    if len(words) >= 12:
        ngrams: dict[tuple, int] = {}
        for i in range(len(words) - 3):
            ng = tuple(words[i:i + 4])
            ngrams[ng] = ngrams.get(ng, 0) + 1
        if ngrams and max(ngrams.values()) >= 3:
            score -= 0.20

    return round(max(0.0, score), 4)


# ──────────────────────────────────────────────────────────
# MAIN SCORER CLASS
# ──────────────────────────────────────────────────────────

class HeuristicScorer:
    """
    Orchestrates all four scoring dimensions into one weighted score.

    Usage:
        scorer  = HeuristicScorer()
        results = scorer.score_all(query, responses)
        winner  = scorer.pick_winner(results)
    """

    def __init__(self, weights: Optional[dict] = None):
        self.weights = weights or WEIGHTS
        total = sum(self.weights.values())
        assert abs(total - 1.0) < 0.01, \
            f"Weights must sum to 1.0, got {total:.4f}"

    def score_one(self, query: str, response: ModelResponse) -> ScoredResponse:
        """Score a single ModelResponse against the query."""

        if not response.success or not response.content:
            return ScoredResponse(
                model_id        = response.model_id,
                content         = response.content or "",
                relevance       = 0.0,
                coherence       = 0.0,
                completeness    = 0.0,
                consistency     = 0.0,
                weighted_score  = 0.0,
                confidence_tier = "LOW",
                latency_ms      = response.latency_ms,
                success         = False,
                error           = response.error,
            )

        r = score_relevance(query, response.content)
        c = score_coherence(response.content)
        k = score_completeness(query, response.content)
        s = score_consistency(response.content)

        weighted = round(
            r * self.weights["relevance"]
            + c * self.weights["coherence"]
            + k * self.weights["completeness"]
            + s * self.weights["consistency"],
            4,
        )

        return ScoredResponse(
            model_id        = response.model_id,
            content         = response.content,
            relevance       = r,
            coherence       = c,
            completeness    = k,
            consistency     = s,
            weighted_score  = weighted,
            confidence_tier = _confidence_tier(weighted),
            latency_ms      = response.latency_ms,
            success         = response.success,
            error           = response.error,
        )

    def score_all(
        self,
        query: str,
        responses: list[ModelResponse],
    ) -> list[ScoredResponse]:
        """Score every response. Failed ones get zeroed out."""
        return [self.score_one(query, r) for r in responses]
