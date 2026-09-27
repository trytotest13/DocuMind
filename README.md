# DocuMind

Chat with your documents and get answers that cite the exact page they came from. Upload a PDF, ask questions, and if the answer is not in the document the app says so instead of making something up.

I kept it to one use case (an HR policy assistant, with a sample company handbook included) so the retrieval quality could actually be measured rather than vibe-checked.

## How it works

A PDF is parsed with PyMuPDF page by page, so page numbers survive into storage. Text is chunked along paragraph boundaries (with a naive fixed-window mode available for comparison), embedded with the default MiniLM ONNX model, and stored in a local ChromaDB collection.

At question time the top chunks are retrieved and passed to the model with a prompt that only allows answers from the provided excerpts, with inline citations in `[source: doc p.N]` form. If the best match is still too far away (cosine distance above 0.9) the model is never called at all and you get a fixed refusal. That check is cheaper than paying for a generation and hoping the model behaves.

## Running it

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
cp .env.example .env    # set OPENAI_API_KEY
.venv/Scripts/python -m uvicorn app.main:app --reload
```

Open http://localhost:8000. The first upload downloads the embedding model (about 80 MB, cached afterwards). Retrieval works fully offline; generation needs the key.

## The eval

There is a 20-question benchmark in `evals/` against the sample handbook, including one out-of-scope question that must trigger the refusal. The sample PDF is generated with fixed page breaks so page accuracy is actually testable.

```
.venv/Scripts/python evals/run_eval.py --retrieval-only   # no API key needed
.venv/Scripts/python evals/run_eval.py                    # full, needs key
.venv/Scripts/python evals/run_eval.py --retrieval-only --strategy fixed   # A/B chunking
```

Current results on the sample document (recursive chunking): 100% retrieval hit rate, 100% correct-page accuracy in the top 3, p50 retrieval latency around 160 ms. The report lands in `evals/eval_report.md`. These numbers are for a 6-page document, so treat them as a sanity check of the pipeline rather than a benchmark of the approach.

## Things I would do next

- Move from Chroma to pgvector so the store lives in Postgres.
- Stream the answers like a normal chat, and keep conversation history.
- Add a reranker between retrieval and generation, which matters more as the document count grows.
