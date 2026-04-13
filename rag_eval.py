"""
RAG Evaluation Script (Improved)
Metrics:
- Answer Correctness
- Faithfulness
- Context Relevance
- Answer Relevance

Uses OpenAI as LLM Judge.
"""

import json
import re
import time
from typing import Optional, List
from openai import OpenAI

# ── Client ────────────────────────────────────────────────────────────────────

client = OpenAI()  # reads OPENAI_API_KEY
MODEL = "gpt-4.1-mini"  # fast + cheap + strong evaluator

# ── Config ────────────────────────────────────────────────────────────────────

MAX_RETRIES = 3
MAX_CONTEXT_CHARS = 10000  # prevent context overflow

# ── Helpers ───────────────────────────────────────────────────────────────────

def _truncate(text: str, max_chars: int) -> str:
    return text[:max_chars] if len(text) > max_chars else text


def _call(system: str, user: str) -> str:
    """Robust OpenAI call with retries + strict JSON output."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,  # deterministic judge
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return str(e)
            time.sleep(1.5 * (attempt + 1))


def _safe_json_load(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _parse_score(text: str) -> float:
    """Fallback score extraction."""
    match = re.search(r"([0-9]*\.?[0-9]+)", text)
    if match:
        val = float(match.group(1))
        return max(0.0, min(1.0, val))
    return 0.0


# ── Metric 1 – Answer Correctness ─────────────────────────────────────────────

def answer_correctness(question: str, answer: str, ground_truth: str) -> dict:
    system = (
        "You are a strict evaluator. Compare generated answer with ground truth.\n"
        "Score:\n"
        "1.0 = fully correct\n"
        "0.5 = partially correct\n"
        "0.0 = incorrect\n"
        "Return JSON with: score (0-1), reason (short)."
    )

    user = f"""
Question:
{question}

Generated Answer:
{answer}

Ground Truth:
{ground_truth}
"""

    raw = _call(system, user)
    data = _safe_json_load(raw)

    score = float(data.get("score", _parse_score(raw)))
    reason = data.get("reason", raw)

    return {
        "metric": "answer_correctness",
        "score": round(score, 4),
        "reason": reason,
    }


# ── Metric 2 – Faithfulness ───────────────────────────────────────────────────

def faithfulness(answer: str, contexts: List[str]) -> dict:
    context_block = "\n\n".join(contexts)
    context_block = _truncate(context_block, MAX_CONTEXT_CHARS)

    system = (
        "You are an expert factuality evaluator.\n"
        "Break the answer into atomic claims.\n"
        "Check if each claim is supported by the context.\n"
        "Return JSON:\n"
        "{total_claims, supported_claims, score (supported/total), reason}"
    )

    user = f"""
Context:
{context_block}

Answer:
{answer}
"""

    raw = _call(system, user)
    data = _safe_json_load(raw)

    total = data.get("total_claims")
    supported = data.get("supported_claims")

    if isinstance(total, int) and total > 0 and isinstance(supported, int):
        score = supported / total
    else:
        score = _parse_score(raw)

    return {
        "metric": "faithfulness",
        "score": round(score, 4),
        "reason": data.get("reason", raw),
        "total_claims": total,
        "supported_claims": supported,
    }


# ── Metric 3 – Context Relevance ──────────────────────────────────────────────

def context_relevance(question: str, contexts: List[str]) -> dict:
    context_block = "\n\n".join(contexts)
    context_block = _truncate(context_block, MAX_CONTEXT_CHARS)

    system = (
        "You are a retrieval evaluator.\n"
        "Estimate how much of the context is useful for answering the question.\n"
        "Return JSON: {score (0-1), reason}"
    )

    user = f"""
Question:
{question}

Context:
{context_block}
"""

    raw = _call(system, user)
    data = _safe_json_load(raw)

    score = float(data.get("score", _parse_score(raw)))

    return {
        "metric": "context_relevance",
        "score": round(score, 4),
        "reason": data.get("reason", raw),
    }


# ── Metric 4 – Answer Relevance ───────────────────────────────────────────────

def answer_relevance(question: str, answer: str) -> dict:
    system = (
        "You are an evaluator.\n"
        "Does the answer directly and completely address the question?\n"
        "Ignore factual correctness.\n"
        "Return JSON: {score (0-1), reason}"
    )

    user = f"""
Question:
{question}

Answer:
{answer}
"""

    raw = _call(system, user)
    data = _safe_json_load(raw)

    score = float(data.get("score", _parse_score(raw)))

    return {
        "metric": "answer_relevance",
        "score": round(score, 4),
        "reason": data.get("reason", raw),
    }


# ── Combined evaluator ─────────────────────────────────────────────────────────

def evaluate(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> dict:

    results = {}

    results["faithfulness"] = faithfulness(answer, contexts)
    results["context_relevance"] = context_relevance(question, contexts)
    results["answer_relevance"] = answer_relevance(question, answer)

    if ground_truth:
        results["answer_correctness"] = answer_correctness(
            question, answer, ground_truth
        )

    scores = [v["score"] for v in results.values()]
    results["overall"] = round(sum(scores) / len(scores), 4)

    return results


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_results(results: dict):
    print("\n" + "=" * 60)
    print("RAG Evaluation Results")
    print("=" * 60)

    for k, v in results.items():
        if k == "overall":
            continue
        print(f"{k:22}: {v['score']:.2f}")
        print(f"  → {v['reason']}\n")

    print(f"{'overall':22}: {results['overall']:.2f}")
    print("=" * 60)


# ── Example ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = {
        "question": "What is the target KB size used for MedMentions?",
        "answer": "It uses the 2017AA version of UMLS.",
        "ground_truth": "The KB size is 2.3M.",
        "contexts": [
            "The KB used is the 2017AA version of UMLS.",
            "The dataset contains 2M entities and 120K mentions.",
        ],
    }

    results = evaluate(**sample)
    print_results(results)
    print(json.dumps(results, indent=2))