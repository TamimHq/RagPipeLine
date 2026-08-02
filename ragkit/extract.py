from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import fitz

_BROKEN = re.compile(r"[\ufffd\u0000-\u0008\u000e-\u001f]")
_LETTERS = re.compile(
    r"[A-Za-z\u0980-\u09FF\u0900-\u097F\u0600-\u06FF\u4e00-\u9fff\u0400-\u04FF]"
)

_DEP = "\u0981-\u0983\u09BE-\u09CC\u09D7\u0900-\u0903\u093E-\u094C"
_VIR = "\u09CD\u094D"
_MALFORMED = re.compile(
    r"(?:^|\s)[" + _DEP + _VIR + r"]"       # dependent mark with nothing to attach to
    r"|[" + _VIR + r"](?=\s|$)"             # virama dangling at word end
)
_INDIC = re.compile(r"[\u0980-\u09FF\u0900-\u097F]")


@dataclass
class PageQuality:
    page: int
    n_chars: int
    letter_ratio: float
    broken_ratio: float
    malformed_ratio: float
    needs_ocr: bool
    reason: str


def malformed_ratio(text: str) -> float:
    if not _INDIC.search(text):
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    return len(_MALFORMED.findall(text)) / len(words)


def page_quality(page: fitz.Page, min_chars: int = 60,
                 min_letter_ratio: float = 0.35,
                 max_broken_ratio: float = 0.02,
                 max_malformed: float = 0.05) -> PageQuality:
    text = unicodedata.normalize("NFC", page.get_text("text") or "")
    n = len(text.strip())

    letter_ratio = len(_LETTERS.findall(text)) / n if n else 0.0
    broken_ratio = len(_BROKEN.findall(text)) / n if n else 0.0
    mal = malformed_ratio(text)

    if n < min_chars:
        needs, why = True, f"only {n} chars in text layer"
    elif broken_ratio > max_broken_ratio:
        needs, why = True, f"{broken_ratio:.1%} replacement/control chars"
    elif mal > max_malformed:
        needs, why = True, f"malformed script ratio {mal:.2f} (mojibake)"
    elif letter_ratio < min_letter_ratio:
        needs, why = True, f"letter ratio {letter_ratio:.2f} too low"
    else:
        needs, why = False, "text layer usable"

    return PageQuality(page.number + 1, n, letter_ratio, broken_ratio, mal, needs, why)


def scan_pdf(path: str, **kw) -> List[PageQuality]:
    doc = fitz.open(path)
    out = [page_quality(p, **kw) for p in doc]
    doc.close()
    return out


def extract_native(path: str, pages: Optional[List[int]] = None) -> Dict[int, str]:
    doc = fitz.open(path)
    want = set(pages) if pages else None
    out = {}
    for p in doc:
        n = p.number + 1
        if want is None or n in want:
            out[n] = unicodedata.normalize("NFC", p.get_text("text") or "")
    doc.close()
    return out


def render_page(path: str, page_no: int, dpi: int = 200) -> bytes:
    doc = fitz.open(path)
    pix = doc[page_no - 1].get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0))
    png = pix.tobytes("png")
    doc.close()
    return png


# --------------------------------------------------------------------------
# Boilerplate discovery — learned per document, not hand-written
# --------------------------------------------------------------------------
def find_boilerplate(pages: Dict[int, str], min_frac: float = 0.5,
                     max_len: int = 90, min_pages: int = 4,
                     max_frac_of_page: float = 0.25) -> List[str]:
    if len(pages) < min_pages:
        return []

    counts: Counter = Counter()
    for text in pages.values():
        seen = set()
        for line in text.split("\n"):
            line = re.sub(r"\s+", " ", line).strip()
            if not line or len(line) > max_len:
                continue
            if re.fullmatch(r"[\d\s.\u09e6-\u09ef\u0660-\u0669-]+", line):
                continue                    # bare page numbers, handled elsewhere
            if line not in seen:
                seen.add(line)
                counts[line] += 1

    threshold = max(min_pages, int(len(pages) * min_frac))
    found = [line for line, c in counts.most_common() if c >= threshold]

    # Safety valve: on documents with highly repetitive pages (forms, tables,
    # slide decks) this rule can match nearly every line and gut the corpus.
    # Cap it at a fraction of the typical page, and bail out entirely if the
    # cap is hit — better to keep some noise than to delete the content.
    avg_lines = sum(len([l for l in t.split("\n") if l.strip()])
                    for t in pages.values()) / max(1, len(pages))
    cap = max(1, int(avg_lines * max_frac_of_page))
    if len(found) > cap:
        return found[:cap]
    return found


def strip_boilerplate(text: str, boilerplate: List[str]) -> str:
    if not boilerplate:
        return text
    bl = set(boilerplate)
    return "\n".join(
        ln for ln in text.split("\n")
        if re.sub(r"\s+", " ", ln).strip() not in bl
    )
