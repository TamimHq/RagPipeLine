from __future__ import annotations

import re
from typing import Dict, List

# Option markers across scripts: (ক) (খ), (a) (b), A., 1), i.
_OPTIONS = [
    re.compile(r"\([কখগঘঙ]\)"),
    re.compile(r"\([a-eA-E]\)"),
    re.compile(r"(?:^|\s)[a-eA-E][.)]\s"),
    re.compile(r"\([ivx]{1,3}\)", re.I),
]
# "answer:" in several languages
_ANSWER = re.compile(r"উত্তর\s*[:ঃ]|answer\s*[:.]|ans\s*[:.]|उत्तर\s*[:ः]", re.I)
# numbered item openers, latin or bengali digits
_NUMBERED = re.compile(r"(?:^|\n)\s*[\d\u09e6-\u09ef]{1,3}\s*[।.)\]]")
_QMARK = re.compile(r"[?？]")
# table-ish: a line with 2+ cell separators
_TABLE_ROW = re.compile(r"^[^|\n]{1,60}(?:\|[^|\n]{1,60}){1,}$", re.M)
_SENT_END = re.compile(r"[।.!?]\s")
# A "bare" table cell: one option letter, a serial number, or a short header
_BARE_CELL = re.compile(r"[কখগঘঙa-eA-E]|[\d\u09e6-\u09ef]{1,3}|SL|Ans|No\.?", re.I)


def _count(patterns, text: str) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def classify(text: str) -> str:
    """Label a single chunk. Cheap, deterministic, no model required."""
    if not text or len(text) < 20:
        return "prose"

    n_lines = max(1, text.count("\n") + 1)
    words = max(1, len(text.split()))

    opts = _count(_OPTIONS, text)
    answers = len(_ANSWER.findall(text))
    numbered = len(_NUMBERED.findall(text))
    qmarks = len(_QMARK.findall(text))
    rows = len(_TABLE_ROW.findall(text))
    sentences = len(_SENT_END.findall(text))
    avg_line = len(text) / n_lines

    # Answer key: many answer markers or option letters, almost no prose.
    if answers >= 3 and sentences <= 2:
        return "answer_key"
    if opts >= 8 and avg_line < 45 and sentences <= 3:
        return "answer_key"

    # Answer-key grid: a table whose cells are mostly bare option letters
    # (ক / খ / a / b) or serial numbers, e.g. the উত্তরমালা page.
    if rows >= 2:
        cells = [c.strip() for row in _TABLE_ROW.findall(text) for c in row.split("|")]
        cells = [c for c in cells if c]
        if cells:
            bare = sum(bool(_BARE_CELL.fullmatch(c)) for c in cells)
            if bare / len(cells) >= 0.7:
                return "answer_key"

    # Question bank: option groups, or numbered items ending in question marks.
    if opts >= 3 or (numbered >= 2 and qmarks >= 2):
        return "question"
    if answers >= 1 and opts >= 2:
        return "question"

    # Reference: table rows, or many short lines with few sentence endings.
    if rows >= 3:
        return "reference"
    if n_lines >= 5 and avg_line < 40 and sentences <= 2:
        return "reference"

    return "prose"


def classify_chunks(chunks: List[Dict]) -> List[Dict]:
    for c in chunks:
        c["section"] = classify(c["text"])
    return chunks


def section_summary(chunks: List[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in chunks:
        out[c["section"]] = out.get(c["section"], 0) + 1
    return out


# Sections that state facts, as opposed to asking about them. Used only for
# reporting and optional prompt hints — NOT as a retrieval quota. A fixed quota
# was measured to cut MRR from 0.93 to 0.29 on exam-style questions, because it
# forced prose into slots the correct answer needed.
ANSWER_BEARING = ("prose", "reference")
