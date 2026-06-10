"""
src/generation/llm_client.py

Generates answers using Groq's LLM API.

The prompt is strictly grounded — the LLM is instructed to answer ONLY
from the provided context, and to return a specific fallback message
when the answer is not present in the document.
"""

from __future__ import annotations

from groq import Groq

import config


# ── prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly 
based on the provided document context.

Rules:
1. Answer ONLY using information from the context below.
2. Do NOT use any external knowledge or make assumptions.
3. If the context does not contain enough information to answer the question, 
   respond with exactly: "The answer is not available in the provided document."
4. Be concise, accurate, and clear.
5. If the question asks for a list, format your answer as a numbered or bulleted list.
"""

USER_PROMPT_TEMPLATE = """Context from the document:
{context}

---

Question: {question}

Answer:"""


# ── fallback detection ────────────────────────────────────────────────────────

FALLBACK_RESPONSE = "The answer is not available in the provided document."

# Minimum cosine similarity score to consider retrieved chunks relevant
MIN_RELEVANCE_SCORE = 0.30


# ── public API ────────────────────────────────────────────────────────────────

def generate_answer(question: str, context: str, top_score: float = 1.0) -> str:
    """
    Generate an answer for `question` grounded in `context`.

    Args:
        question:   User's question string.
        context:    Concatenated retrieved chunks.
        top_score:  Cosine similarity score of the best-matching chunk.
                    If below MIN_RELEVANCE_SCORE, skip LLM and return fallback.

    Returns:
        Answer string.
    """
    # If retrieval confidence is too low, skip LLM call entirely
    if top_score < MIN_RELEVANCE_SCORE:
        print(
            f"[LLM] Top retrieval score {top_score} < threshold {MIN_RELEVANCE_SCORE}. "
            "Returning fallback."
        )
        return FALLBACK_RESPONSE

    client = Groq(api_key=config.GROQ_API_KEY)

    user_message = USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=config.GROQ_MAX_TOKENS,
        temperature=config.GROQ_TEMPERATURE,
    )

    answer: str = response.choices[0].message.content.strip()

    print(f"[LLM] Generated answer ({len(answer)} chars).")
    return answer