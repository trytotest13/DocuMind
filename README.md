# DocuMind — Chat with Your Documents (RAG, with citations + evals)

**The skill every AI job post lists.** Upload a PDF → it's chunked, embedded and stored in a vector DB → ask questions and get answers **with page-level citations** — plus a refusal when the docs don't contain the answer. Includes a 20-question eval harness so you can quote real numbers.

Pick a vertical to avoid looking like a toy: HR policy assistant (sample doc included), syllabus Q&A, or a support bot for a real product's docs.

## Features

- **Ingestion** — PDF (PyMuPDF, page numbers preserved) / TXT / MD → paragraph-aware recursive chunking → embeddings → **ChromaDB** (persistent, local, no server needed)
- **Two chunking strategies** (`recursive` vs `fixed` baseline) selectable at upload and in the eval — do the A/B comparison and put it in the README
- **Grounded generation** — system prompt forbids outside knowledge, requires inline `[source: doc p.N]` citations
- **Refusal rule** — if the best retrieved chunk is too dissimilar (cosine distance > threshold), the LLM is never called: cheaper and safer than hoping the model behaves
- **Eval harness** — retrieval hit rate, correct-page accuracy, answer keyword accuracy, refusal correctness, p50 latency → writes `evals/eval_report.md`

## Architecture

```mermaid
flowchart LR
  PDF[Upload PDF/TXT/MD] --> X[PyMuPDF extract<br/>page-numbered text]
  X --> C[Chunking<br/>recursive / fixed A/B]
  C --> E[all-MiniLM-L6-v2<br/>ONNX embeddings]
  E --> V[(ChromaDB<br/>cosine index)]
  Q[Question] --> R[Top-k retrieval]
  V --> R
  R -->|"distance > 0.9 → refuse<br/>(no LLM call)"| D{Grounded?}
  R --> G[LLM w/ strict<br/>grounded prompt]
  G --> A[Answer + [source: doc p.N] citations]
```

## Run it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows (Linux/macOS: .venv/bin/pip)
cp .env.example .env                               # set OPENAI_API_KEY (+ BASE_URL / LLM_MODEL)
.venv/Scripts/python -m uvicorn app.main:app --reload
# open http://localhost:8000
```

First upload downloads the embedding model (~80 MB ONNX, cached after that). Fully works offline for retrieval; generation needs the API key.

## Run the eval (the interview gold)

```bash
# retrieval-only, zero API key needed:
.venv/Scripts/python evals/run_eval.py --retrieval-only
# full eval (needs .env with API key):
.venv/Scripts/python evals/run_eval.py
# A/B the chunking strategies:
.venv/Scripts/python evals/run_eval.py --retrieval-only --strategy fixed
```

Sample doc (`data/sample/company_handbook.txt`) has known page locations for all 20 questions, including one out-of-scope question that must trigger the refusal.

## Resume bullet

> Built a RAG document assistant with page-level citations using ChromaDB + a distance-threshold refusal rule; evaluated on a 20-question benchmark — X% retrieval hit rate, Y% answer accuracy, p50 retrieval Z ms; A/B-tested recursive vs fixed chunking.

## Roadmap

- [ ] Swap Chroma → **pgvector** (Neon free tier) — the JD keyword; keep Chroma as the fallback
- [ ] Streaming answers (SSE) + conversation memory
- [ ] Reranker (cross-encoder) for top-20 → top-5
- [ ] Deploy: Render/Railway (FastAPI) + Vercel (Next.js frontend upgrade)
