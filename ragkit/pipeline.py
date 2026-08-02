from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import time
import unicodedata
from typing import Callable, Dict, List, Optional

from extract import (extract_native, find_boilerplate, render_page, scan_pdf,
                     strip_boilerplate)
from index import chunk_document
from sections import classify_chunks, section_summary

OCR_PROMPT = """Transcribe ALL text in this page image exactly as printed.

Rules:
- Be character-accurate. For Indic scripts pay close attention to conjuncts
  (ম্ভ, ন্ত, ষ্ট, ক্ষ, স্ত, ঙ্গ, দ্ধ) - do not substitute similar-looking ones.
- Preserve reading order, line breaks, and blank lines between paragraphs.
- For tables, output one row per line with cells separated by " | ".
- Do NOT translate, summarize, correct, or explain.
- Output the transcription only - no preamble, no markdown fences."""


def file_id(path: str) -> str:
    """Stable per-file key: name plus a hash of contents, so edits invalidate."""
    h = hashlib.md5(open(path, "rb").read()).hexdigest()[:8]
    return f"{os.path.splitext(os.path.basename(path))[0]}_{h}"


def clean_text(text: str, boilerplate: List[str]) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = strip_boilerplate(text, boilerplate)
    out = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t\u00a0]+", " ", line).strip()
        if not line:
            out.append("")
            continue
        if re.fullmatch(r"[\d\u09e6-\u09ef\u0660-\u0669]{1,4}", line):
            continue                      # bare page number
        if not re.search(r"[\w\u0980-\u09FF\u0900-\u097F\u0600-\u06FF]", line):
            continue                      # rules, dingbats, OCR noise
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def ingest_pdf(path: str, ocr_fn: Optional[Callable[[bytes], str]] = None,
               cache_dir: str = "./cache", dpi: int = 200,
               force_ocr: bool = False, verbose: bool = True) -> Dict[int, str]:
    """Return {page_no: text}, OCR'ing only the pages that need it."""
    doc_id = file_id(path)
    pdir = os.path.join(cache_dir, doc_id)
    os.makedirs(pdir, exist_ok=True)

    quality = scan_pdf(path)
    need = [q.page for q in quality if (force_ocr or q.needs_ocr)]
    good = [q.page for q in quality if q.page not in need]

    if verbose:
        why = {}
        for q in quality:
            if q.page in need:
                why[q.reason.split("(")[0].strip()] = why.get(
                    q.reason.split("(")[0].strip(), 0) + 1
        print(f"  {os.path.basename(path)}: {len(quality)} pages, "
              f"{len(need)} need OCR {why if why else ''}")

    pages: Dict[int, str] = {}
    pages.update(extract_native(path, good))

    for p in need:
        cache = os.path.join(pdir, f"p{p:04d}.txt")
        if os.path.exists(cache):
            pages[p] = open(cache, encoding="utf-8").read()
            continue
        if ocr_fn is None:
            pages[p] = ""                 # no OCR available; skip rather than poison
            continue
        text = ""
        png = render_page(path, p, dpi)
        for attempt in range(3):
            try:
                text = ocr_fn(png) or ""
                break
            except Exception as e:
                print(f"    p{p} OCR attempt {attempt+1}: {e}")
                time.sleep(2 * (attempt + 1))
        open(cache, "w", encoding="utf-8").write(text)
        pages[p] = text
        time.sleep(0.4)
        if verbose and p % 10 == 0:
            print(f"    OCR'd through page {p}")

    return pages


def build_corpus(folder: str, index, ocr_fn=None, cache_dir: str = "./cache",
                 pattern: str = "*.pdf", **chunk_kw) -> List[Dict]:
    """Ingest every PDF in a folder into one shared index."""
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        print(f"No files matching {pattern} in {folder}")
        return []

    all_chunks = []
    for path in paths:
        doc_id = file_id(path)
        pages = ingest_pdf(path, ocr_fn, cache_dir)
        bp = find_boilerplate(pages)
        if bp:
            print(f"    boilerplate learned: {bp[:3]}")
        cleaned = {p: clean_text(t, bp) for p, t in pages.items()}
        chunks = chunk_document(cleaned, doc_id,
                                chunk_size=index.cfg["chunk_size"],
                                chunk_overlap=index.cfg["chunk_overlap"],
                                **chunk_kw)
        classify_chunks(chunks)
        print(f"    {len(chunks)} chunks {section_summary(chunks)}")
        index.add(chunks, cache_key=doc_id)
        all_chunks.extend(chunks)

    print(f"\nIndexed {len(all_chunks)} chunks from {len(paths)} documents.")
    print(f"Documents: {index.docs}")
    return all_chunks


# --------------------------------------------------------------------------
ANSWER_PROMPT = """Answer the question using only the context below.

Rules:
- If the context does not contain the answer, say exactly: NOT_FOUND
- Context may include quiz questions and option lists. Those are not facts —
  answer from explanatory or narrative passages instead.
- Narrative may use pronouns or first person; resolve them when the referent is
  clear from the context, and say who you mean.
- Answer in the same language as the question. Do not mix languages.
- Be concise. Plain text, no markdown.

Context:
{context}

Question: {query}

Answer:"""


class RAG:
    def __init__(self, index, generate: Callable[[str], str],
                 prompt: str = ANSWER_PROMPT, not_found: str = "NOT_FOUND"):
        self.index = index
        self.generate = generate
        self.prompt = prompt
        self.not_found = not_found
        self.history: List[Dict] = []

    def ask(self, query: str, docs=None, debug: bool = False,
            return_sources: bool = False):
        chunks = self.index.retrieve(query, docs=docs, debug=debug)
        if not chunks:
            return (self.not_found, []) if return_sources else self.not_found
        ctx = "\n\n---\n\n".join(
            f"[{c['doc']} p{c['page']}]\n{c['text']}" for c in chunks)
        try:
            ans = (self.generate(self.prompt.format(context=ctx, query=query)) or "").strip()
        except Exception as e:
            ans = f"LLM error: {e}"
        self.history.append({"q": query, "a": ans,
                             "sources": [(c["doc"], c["page"]) for c in chunks]})
        return (ans, chunks) if return_sources else ans
