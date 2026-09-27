"""Grounded generation: answer ONLY from the retrieved context, with citations.

Refusal rule: if the best retrieved chunk is still too dissimilar (cosine distance
above the threshold), the model is not called at all: we refuse without spending
tokens, which is both cheaper and safer than hoping the LLM behaves.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI

from .ingest import RetrievedChunk, retrieve

load_dotenv()

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

MAX_DISTANCE = 0.9  # cosine distance threshold for refusal (tuned on the eval set)

SYSTEM_PROMPT = """You are DocuMind, a careful assistant that answers questions using ONLY the provided document excerpts.

Rules:
1. Answer only from the excerpts. Never use outside knowledge.
2. Cite every claim inline as [source: <doc_name> p.<page>].
3. If the excerpts do not contain the answer, reply exactly:
   "I couldn't find this in your documents."
4. Be concise."""


class LlmNotConfigured(RuntimeError):
    pass


@dataclass
class Answer:
    text: str
    citations: list[dict] = field(default_factory=list)
    refused: bool = False
    latency_ms: int = 0


def _client() -> OpenAI:
    if not API_KEY:
        raise LlmNotConfigured("Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL / LLM_MODEL) in .env")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def _format_context(hits: list[RetrievedChunk]) -> str:
    parts = []
    for h in hits:
        page = f"p.{h.page}" if h.page else "document"
        parts.append(f"[source: {h.doc_name} {page}]\n{h.text}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, doc_id: str | None = None) -> Answer:
    t0 = time.perf_counter()
    hits = retrieve(question, doc_id=doc_id)
    citations = [
        {"doc_name": h.doc_name, "page": h.page, "snippet": h.text[:160], "distance": round(h.score, 4)}
        for h in hits
    ]

    if not hits or hits[0].score > MAX_DISTANCE:
        return Answer(
            text="I couldn't find this in your documents.",
            citations=citations,
            refused=True,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    prompt = f"Document excerpts:\n\n{_format_context(hits)}\n\nQuestion: {question}"
    client = _client()
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=700,
    )
    text = (res.choices[0].message.content or "").strip()
    return Answer(text=text, citations=citations, latency_ms=int((time.perf_counter() - t0) * 1000))
