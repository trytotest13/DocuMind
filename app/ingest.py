"""Document ingestion: extract text (with page numbers) -> chunk -> embed -> store in Chroma.

Two chunking strategies are supported so you can A/B them in the eval:
  - "recursive" (default): split on paragraph boundaries first, then hard-split long ones
  - "fixed":     naive fixed-size character windows (baseline)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (new import name; old `import fitz` is deprecated)
import chromadb
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "5"))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
# Default embedding: all-MiniLM-L6-v2 via ONNX (downloaded once, ~80 MB): no GPU, no torch.
collection = client.get_or_create_collection(name="documind", metadata={"hnsw:space": "cosine"})


@dataclass
class Chunk:
    text: str
    page: int  # 1-based; 0 for non-PDF docs
    index: int


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)]: page numbers preserved for citations."""
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        pages = [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
        doc.close()
        return pages
    return [(0, path.read_text(encoding="utf-8", errors="replace"))]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)] if text else []


def chunk_pages(pages: list[tuple[int, str]], strategy: str = "recursive") -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for page_no, text in pages:
        text = re.sub(r"[ \t]+", " ", text).strip()
        if not text:
            continue
        if strategy == "recursive":
            # paragraph-aware: keep paragraphs whole when possible
            paragraphs = re.split(r"\n\s*\n", text)
            buf = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(para) > CHUNK_SIZE:
                    if buf:
                        chunks.append(Chunk(buf, page_no, idx)); idx += 1
                        buf = ""
                    for piece in _hard_split(para, CHUNK_SIZE, CHUNK_OVERLAP):
                        chunks.append(Chunk(piece, page_no, idx)); idx += 1
                    continue
                if len(buf) + len(para) + 2 > CHUNK_SIZE:
                    chunks.append(Chunk(buf, page_no, idx)); idx += 1
                    buf = para
                else:
                    buf = f"{buf}\n\n{para}" if buf else para
            if buf:
                chunks.append(Chunk(buf, page_no, idx)); idx += 1
        else:  # fixed baseline
            for piece in _hard_split(text, CHUNK_SIZE, CHUNK_OVERLAP):
                chunks.append(Chunk(piece, page_no, idx)); idx += 1
    return chunks


def ingest_file(path: Path, strategy: str = "recursive") -> dict:
    """Ingest one document; returns stats. Re-ingesting the same file replaces it."""
    doc_id = path.stem
    delete_doc(doc_id)

    pages = extract_pages(path)
    chunks = chunk_pages(pages, strategy=strategy)
    if not chunks:
        raise ValueError(f"No extractable text found in {path.name}")

    collection.add(
        ids=[f"{doc_id}:{c.index}" for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"doc_id": doc_id, "doc_name": path.name, "page": c.page, "chunk_index": c.index, "strategy": strategy} for c in chunks],
    )
    return {"doc_id": doc_id, "doc_name": path.name, "pages": len(pages), "chunks": len(chunks), "strategy": strategy}


def delete_doc(doc_id: str) -> None:
    try:
        collection.delete(where={"doc_id": doc_id})
    except Exception:
        pass


@dataclass
class RetrievedChunk:
    text: str
    doc_name: str
    page: int
    score: float  # cosine distance: lower is better


def retrieve(question: str, doc_id: str | None = None, k: int = TOP_K) -> list[RetrievedChunk]:
    where = {"doc_id": doc_id} if doc_id else None
    res = collection.query(query_texts=[question], n_results=min(k, max(1, collection.count())), where=where)
    hits: list[RetrievedChunk] = []
    docs = res.get("documents") or [[]]
    metas = res.get("metadatas") or [[]]
    dists = res.get("distances") or [[]]
    for text, meta, dist in zip(docs[0], metas[0], dists[0]):
        hits.append(RetrievedChunk(text=text, doc_name=meta["doc_name"], page=int(meta.get("page", 0)), score=float(dist)))
    return hits
