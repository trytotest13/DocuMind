"""Generate data/sample/company_handbook.pdf with controlled page breaks so the
eval can check page-level citations:

  page 1: title page          page 4: sick leave + remote work
  page 2: working hours       page 5: expenses + conduct & security
  page 3: vacation            page 6: performance & promotion

Run directly, or let run_eval.py call build_if_missing().
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "company_handbook.pdf"

PAGES: dict[int, list[str]] = {
    1: ["Company Handbook", "Northwind Labs (Sample Document for Testing)", "", "Version 3.2 — internal use only"],
    2: [
        "1. WORKING HOURS AND ATTENDANCE",
        "Standard working hours are 9:30 AM to 6:00 PM local time, Monday through",
        "Friday. Employees working from home must remain reachable during core",
        "hours (11:00 AM to 4:00 PM). Attendance is tracked via the team calendar;",
        "repeated unexcused absences are handled by your manager and HR together.",
    ],
    3: [
        "2. VACATION POLICY",
        "Full-time employees accrue 20 days of paid vacation per year, accrued at",
        "1.67 days per month. New employees may take vacation after completing 90",
        "days of employment. Up to 5 unused vacation days may be carried over into",
        "the next calendar year; anything above 5 days is forfeited on January 1st.",
        "Vacation requests of 5 or more consecutive days require manager approval",
        "at least 2 weeks in advance. Public holidays are additional to vacation days.",
    ],
    4: [
        "3. SICK LEAVE",
        "Employees receive 10 paid sick days per year. A medical certificate is",
        "required for absences of 3 or more consecutive days. Sick leave does not",
        "carry over between years. Extended illness beyond 10 days moves to",
        "short-term disability coverage after HR review.",
        "",
        "4. REMOTE WORK",
        "Employees may work remotely up to 3 days per week by default; fully",
        "remote arrangements require VP approval. Remote employees outside",
        "commuting distance receive a monthly remote work stipend of $150 to",
        "cover internet and utilities. Company equipment (laptop, monitor) is",
        "provided by IT and must be returned on exit.",
    ],
    5: [
        "5. EXPENSES",
        "Daily meal allowance while traveling is $60. Flights over 8 hours may be",
        "booked in premium economy. Expense reports must be submitted within 30",
        "days of purchase. Expense reports over $500 in total require approval",
        "from the finance director, before submission; smaller reports are",
        "approved by direct managers. Alcohol is not reimbursable.",
        "",
        "6. CONDUCT AND SECURITY",
        "Confidential documents must not be shared outside the company. Report",
        "suspected phishing to security@northwind-labs.example within 24 hours.",
        "Passwords must be rotated every 90 days and must be at least 12",
        "characters. Employees may not use personal USB drives on company machines.",
    ],
    6: [
        "7. PERFORMANCE AND PROMOTION",
        "Performance reviews happen twice a year, in March and September.",
        "Promotions require two consecutive \"meets expectations\" reviews or one",
        "\"exceeds expectations\". Employees on a performance improvement plan are",
        "not eligible for promotion during the plan period.",
    ],
}


def build() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for page_no in sorted(PAGES):
        page = doc.new_page()
        y = 72
        for line in PAGES[page_no]:
            page.insert_text((72, y), line, fontsize=12, fontname="helv")
            y += 20
    doc.save(OUT)
    doc.close()
    return OUT


def build_if_missing() -> Path:
    if not OUT.exists():
        build()
    return OUT


if __name__ == "__main__":
    print(f"Wrote {build()}")
