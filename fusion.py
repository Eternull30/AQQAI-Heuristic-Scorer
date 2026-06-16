"""
AQQAI — Response Fusion Engine (fusion.py)
==========================================
Combines responses from all 3 models into one superior response.
No extra API calls — uses sentence-transformers (already installed).

How it works:
  1. Split every model response into individual sentences
  2. Score each sentence against the query using sentence-transformers
  3. Weight each sentence by both its relevance AND its model's overall score
  4. Remove duplicate/near-duplicate sentences (cosine similarity > 0.85)
  5. Pick the top N sentences
  6. Re-order them logically (intro → body → conclusion)
  7. Join into one clean fused response

Why this approach:
  - No extra API call — fully offline after model responses collected
  - Uses sentence-transformers already installed for coherence scoring
  - Each model contributes its best sentences — not just one model wins
  - Deduplication ensures no repetition even if models say same thing
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from scorer import ScoredResponse

# Reuse the same model instance already loaded in scorer.py
# If scorer loaded it, we get it from cache — no re-download
_ST_MODEL = None

def _get_model() -> SentenceTransformer:
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL


def _split_sentences(text: str) -> list[str]:
    """Split text into clean sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 20]


def _is_structural(sentence: str) -> bool:
    """
    Detect structural/header sentences that don't add content.
    e.g. "Here's a simple breakdown:", "Let me explain:", "Sure!"
    These come from markdown headers or intro filler — skip them.
    """
    sentence = sentence.strip()
    structural_patterns = [
        r"^(sure|certainly|absolutely|of course|great)[!.,]?$",
        r"^here'?s?\s+(a|an|the|my|some)",
        r"^let me (explain|break|walk|show)",
        r"^in (summary|conclusion|short|brief)",
        r"^\d+\.",          # numbered list starters like "1."
        r"^#+\s",           # markdown headers
        r"^[-*•]\s",        # bullet points
    ]
    lower = sentence.lower()
    for pattern in structural_patterns:
        if re.match(pattern, lower):
            return True
    # Very short sentences are usually filler
    if len(sentence.split()) < 5:
        return True
    return False


def fuse_responses(query: str, scored_responses: list[ScoredResponse]) -> str:
    """
    Main fusion function.
    Takes all scored responses, returns one fused response string.

    Args:
        query:            The original user query
        scored_responses: List of ScoredResponse from HeuristicScorer

    Returns:
        A single fused response string combining the best of all models.
        Falls back to highest scoring response if fusion fails.
    """
    model = _get_model()

    # ── Step 1: Collect all sentences from successful responses ──
    successful = [r for r in scored_responses if r.success and r.content.strip()]

    if not successful:
        return ""

    if len(successful) == 1:
        # Only one model succeeded — return it directly
        return successful[0].content

    all_sentences = []
    for r in successful:
        sentences = _split_sentences(r.content)
        for sent in sentences:
            if not _is_structural(sent):
                all_sentences.append({
                    "text":        sent,
                    "model_id":    r.model_id,
                    "model_score": r.weighted_score,
                })

    if not all_sentences:
        # Fallback — return best scoring response
        return max(successful, key=lambda r: r.weighted_score).content

    # ── Step 2: Score each sentence against the query ────────────
    query_embedding    = model.encode([query])
    sentence_texts     = [s["text"] for s in all_sentences]
    sentence_embeddings = model.encode(sentence_texts)

    for i, sent in enumerate(all_sentences):
        query_sim = float(
            cosine_similarity([query_embedding[0]], [sentence_embeddings[i]])[0][0]
        )
        # Final sentence score:
        # 60% — how relevant is this sentence to the query
        # 40% — how good was the model that produced it overall
        sent["sentence_score"] = (query_sim * 0.6) + (sent["model_score"] * 0.4)

    # ── Step 3: Remove near-duplicate sentences ──────────────────
    # Sort by score descending — best sentences first
    sorted_sents = sorted(all_sentences, key=lambda x: x["sentence_score"], reverse=True)

    selected         = []
    selected_embeddings = []

    for sent in sorted_sents:
        idx = sentence_texts.index(sent["text"])
        emb = sentence_embeddings[idx]

        if selected_embeddings:
            # Check similarity against all already selected sentences
            sims = cosine_similarity([emb], selected_embeddings)[0]
            if max(sims) > 0.82:
                # Too similar to something already selected — skip
                continue

        selected.append(sent)
        selected_embeddings.append(emb)

        # Cap at 10 sentences — enough for a complete response
        if len(selected) >= 10:
            break

    if not selected:
        return max(successful, key=lambda r: r.weighted_score).content

    # ── Step 4: Re-order sentences logically ─────────────────────
    # Strategy:
    #   - Definition/intro sentences first (contain "is", "are", "refers to")
    #   - Explanation sentences in the middle
    #   - Example/use case sentences after
    #   - Conclusion sentences last

    def _sentence_order_score(sent_text: str) -> int:
        lower = sent_text.lower()
        if any(p in lower for p in ["is a", "is an", "are a", "refers to", "defined as", "known as"]):
            return 0   # definition → goes first
        if any(p in lower for p in ["for example", "for instance", "such as", "e.g", "like"]):
            return 2   # example → goes after explanation
        if any(p in lower for p in ["use case", "used for", "application", "benefit", "advantage"]):
            return 3   # use cases → near end
        if any(p in lower for p in ["in summary", "in conclusion", "overall", "to summarize"]):
            return 4   # conclusion → goes last
        return 1       # explanation → goes in middle

    selected_sorted = sorted(selected, key=lambda s: _sentence_order_score(s["text"]))

    # ── Step 5: Join into final response ─────────────────────────
    fused = " ".join(s["text"] for s in selected_sorted)

    return fused.strip()