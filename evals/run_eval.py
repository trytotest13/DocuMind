"""RAG eval harness.

Two modes:
  python evals/run_eval.py --retrieval-only   # no API key needed: retrieval hit rate + page accuracy
  python evals/run_eval.py                    # full: + answer keyword accuracy + refusal check

Writes evals/eval_report.md with the numbers you can quote on your resume.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import ingest
from app.ingest import ingest_file, retrieve  # noqa: E402
from evals.make_sample_pdf import build_if_missing  # noqa: E402

QUESTIONS = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))["questions"]
SAMPLE_DOC = build_if_missing()  # data/sample/company_handbook.pdf (page layout matches questions.json)


def run_retrieval(questions: list[dict]) -> dict:
    hits = 0
    page_hits = 0
    latencies = []
    failures = []
    for q in questions:
        t0 = time.perf_counter()
        results = retrieve(q["question"], doc_id="company_handbook", k=5)
        latencies.append((time.perf_counter() - t0) * 1000)

        top_pages = [r.page for r in results[:3]]
        got_page = q["page"] in top_pages if q["page"] else True

        if results:
            hits += 1
        if got_page:
            page_hits += 1
        else:
            failures.append({"id": q["id"], "question": q["question"], "expected_page": q["page"], "got_pages": top_pages})

    n = len(questions)
    return {
        "n": n,
        "retrieval_hit_rate": round(hits / n * 100, 1),
        "page_accuracy": round(page_hits / n * 100, 1),
        "p50_ms": round(sorted(latencies)[n // 2], 1),
        "failures": failures,
    }


def run_full(questions: list[dict]) -> dict:
    from app.generate import answer_question, LlmNotConfigured

    keyword_ok = 0
    refusal_ok = 0
    llm_n = 0
    answer_failures = []
    for q in questions:
        try:
            ans = answer_question(q["question"], doc_id="company_handbook")
        except LlmNotConfigured:
            print("LLM not configured — run with --retrieval-only or set .env")
            return run_retrieval(questions)
        llm_n += 1
        text = ans.text.lower()
        if q.get("refusal_expected"):
            if ans.refused or "couldn't find" in text:
                refusal_ok += 1
            else:
                answer_failures.append({"id": q["id"], "question": q["question"], "issue": "should have refused", "answer": ans.text[:120]})
            continue
        missing = [kw for kw in q["must_contain"] if kw.lower() not in text]
        if not missing:
            keyword_ok += 1
        else:
            answer_failures.append({"id": q["id"], "question": q["question"], "missing": missing, "answer": ans.text[:120]})

    base = run_retrieval(questions)
    answerable = sum(1 for q in questions if not q.get("refusal_expected"))
    base["answer_accuracy"] = round(keyword_ok / answerable * 100, 1) if answerable else None
    base["refusal_correct"] = f"{refusal_ok}/1"
    base["answer_failures"] = answer_failures
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--strategy", default="recursive", choices=["recursive", "fixed"])
    args = parser.parse_args()

    # always re-ingest so the chosen chunking strategy is what gets scored
    print(f"Ingesting sample doc (strategy={args.strategy})…")
    ingest_file(SAMPLE_DOC, strategy=args.strategy)

    print("Running eval…")
    report = run_full(QUESTIONS) if not args.retrieval_only else run_retrieval(QUESTIONS)
    report["strategy"] = args.strategy

    out = Path(__file__).parent / "eval_report.md"
    failures_md = "\n".join(
        f"- Q{f['id']} “{f['question']}” → expected p.{f.get('expected_page')}, got {f.get('got_pages')}"
        if "expected_page" in f
        else f"- Q{f['id']} “{f['question']}” → missing {f.get('missing', f.get('issue'))}: “{f.get('answer', '')[:80]}…”"
        for f in report.get("failures", []) + report.get("answer_failures", [])
    ) or "None"
    out.write_text(
        "# DocuMind eval report\n\n"
        f"- Chunking strategy: **{report['strategy']}**\n"
        f"- Retrieval hit rate: **{report['retrieval_hit_rate']}%** ({report['n']} questions)\n"
        f"- Correct-page accuracy (top-3): **{report['page_accuracy']}%**\n"
        + (f"- Answer keyword accuracy: **{report['answer_accuracy']}%**\n" if report.get("answer_accuracy") is not None else "")
        + (f"- Refusal on out-of-scope question: **{report.get('refusal_correct')}**\n" if report.get("refusal_correct") else "")
        + f"- p50 retrieval latency: **{report['p50_ms']} ms**\n\n"
        f"## Failures\n{failures_md}\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if "failures" not in k}, indent=2))
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
