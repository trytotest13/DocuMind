"""DocuMind — chat with your documents, with page-level citations.

Endpoints:
  GET  /                 chat UI
  POST /upload           ingest a PDF/TXT/MD file (multipart: file, strategy=recursive|fixed)
  POST /chat             {"question": "...", "doc_id": null}  -> {answer, citations, latency_ms, refused}
  GET  /documents        list ingested docs
  DELETE /documents/{id}
  GET  /health
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from . import generate, ingest

load_dotenv()

app = FastAPI(title="DocuMind RAG", version="0.1.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "chunks": ingest.collection.count(), "llm_configured": bool(generate.API_KEY)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/upload")
async def upload(file: UploadFile = File(...), strategy: str = Form("recursive")) -> dict:
    if strategy not in ("recursive", "fixed"):
        raise HTTPException(400, "strategy must be 'recursive' or 'fixed'")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type {suffix!r} — use PDF, TXT or MD")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 20 MB)")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / (file.filename or "upload.txt")
        path.write_bytes(data)
        try:
            stats = ingest.ingest_file(path, strategy=strategy)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return stats


@app.post("/chat")
def chat(question: str, doc_id: str | None = None) -> dict:
    question = question.strip()
    if not question:
        raise HTTPException(400, "question is required")
    try:
        ans = generate.answer_question(question, doc_id=doc_id)
    except generate.LlmNotConfigured as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # provider errors -> 502, keep message short
        raise HTTPException(502, f"LLM provider error: {e}")
    return {
        "answer": ans.text,
        "citations": ans.citations,
        "refused": ans.refused,
        "latency_ms": ans.latency_ms,
    }


@app.get("/documents")
def documents() -> dict:
    seen: dict[str, dict] = {}
    for meta in ingest.collection.get(include=["metadatas"])["metadatas"]:
        doc_id = meta["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = {"doc_id": doc_id, "doc_name": meta["doc_name"], "chunks": 0, "strategy": meta.get("strategy")}
        seen[doc_id]["chunks"] += 1
    return {"documents": list(seen.values())}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    ingest.delete_doc(doc_id)
    return {"ok": True}
