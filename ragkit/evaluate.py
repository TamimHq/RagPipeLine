from __future__ import annotations

import copy
import json
import random
import re
import unicodedata
from typing import Callable, Dict, List, Optional

import numpy as np

_PUNCT = re.compile(r"[^\w\u0980-\u09FF\u0900-\u097F\s]", re.UNICODE)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", s)).strip().lower()


def toks(s: str) -> List[str]:
    return [t for t in norm(s).split() if len(t) > 1]


def matches(gold: Dict, text: str, thresh: float = 0.75) -> bool:
    if gold.get("regex"):
        return bool(re.search(gold["regex"], unicodedata.normalize("NFC", text or "")))
    gt = toks(gold.get("answer", ""))
    if not gt:
        return False
    have = set(toks(text))
    return sum(t in have for t in gt) / len(gt) >= thresh


# --------------------------------------------------------------------------
# Gold construction
# --------------------------------------------------------------------------
_MCQ = re.compile(
    r"[০-৯\d]+[।.]\s*(?P<q>[^\n]{8,140}?)\s*"
    r"\([কa]\)\s*(?P<a>[^\n()]{1,60}?)\s*\([খb]\)\s*(?P<b>[^\n()]{1,60}?)\s*"
    r"\([গc]\)\s*(?P<c>[^\n()]{1,60}?)\s*\([ঘd]\)\s*(?P<d>[^\n()]{1,60}?)\s*"
    r"(?:উত্তর|answer|ans)\s*[:ঃ.]\s*(?P<key>[কখগঘabcd])", re.S | re.I)

_KEYMAP = {"ক": 0, "খ": 1, "গ": 2, "ঘ": 3, "a": 0, "b": 1, "c": 2, "d": 3}


def gold_from_answer_keys(chunks: List[Dict], limit: int = 60) -> List[Dict]:
    out, seen = [], set()
    for c in chunks:
        for m in _MCQ.finditer(c["text"]):
            opts = [m.group("a"), m.group("b"), m.group("c"), m.group("d")]
            ans = opts[_KEYMAP[m.group("key").lower()]].strip(" ।.")
            q = re.sub(r"\s+", " ", m.group("q")).strip()
            if len(ans) < 3 or len(q) < 10 or q in seen:
                continue
            seen.add(q)
            out.append({"q": q, "answer": ans, "regex": None,
                        "doc": c.get("doc"), "page": c.get("page"),
                        "source": "answer_key"})
    return out[:limit]


LLM_GOLD_PROMPT = """From the passage below, write {n} factual question-answer pairs.

Rules:
- The answer must be a short span copied verbatim from the passage (1-6 words).
- Ask about specific facts: names, numbers, places, relationships.
- Do not ask anything answerable without reading this passage.
- Write the question in the same language as the passage.
- Output ONLY a JSON array: [{{"q": "...", "answer": "..."}}]

Passage:
{passage}"""


def gold_from_llm(chunks: List[Dict], generate: Callable[[str], str],
                  n_chunks: int = 15, per_chunk: int = 2,
                  sections=("prose",), seed: int = 0) -> List[Dict]:
    pool = [c for c in chunks if not sections or c.get("section") in sections]
    pool = [c for c in pool if len(c["text"]) > 200]
    if not pool:
        pool = [c for c in chunks if len(c["text"]) > 200]
    random.Random(seed).shuffle(pool)

    out = []
    for c in pool[:n_chunks]:
        raw = generate(LLM_GOLD_PROMPT.format(n=per_chunk, passage=c["text"][:1800]))
        raw = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
        try:
            for item in json.loads(raw):
                q, a = str(item.get("q", "")).strip(), str(item.get("answer", "")).strip()
                if len(q) > 10 and 1 <= len(a.split()) <= 8 and norm(a) in norm(c["text"]):
                    out.append({"q": q, "answer": a, "regex": None,
                                "doc": c.get("doc"), "page": c.get("page"),
                                "source": "llm", "gold_chunk": c["id"]})
        except Exception:
            continue
    return out


def target_counts(gold: List[Dict], corpus: List[Dict]) -> List[int]:
    return [sum(matches(g, c["text"]) for c in corpus) for g in gold]


def filter_gold(gold: List[Dict], corpus: List[Dict],
                max_targets: int = 3, verbose: bool = True) -> List[Dict]:
    """Keep only questions whose answer is rare enough to require real ranking."""
    counts = target_counts(gold, corpus)
    kept = [g for g, n in zip(gold, counts) if 1 <= n <= max_targets]
    if verbose:
        med = int(np.median(counts)) if counts else 0
        print(f"gold filter: {len(kept)}/{len(gold)} kept "
              f"(<= {max_targets} targets; median targets was {med})")
    return kept


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def random_baseline(index, gold: List[Dict], top_k: Optional[int] = None,
                    trials: int = 200, seed: int = 0) -> float:
    """Chance hit@k on this corpus. Without it, an accuracy number is unreadable."""
    rng = random.Random(seed)
    top_k = top_k or index.cfg["top_k"]
    if not index.corpus:
        return 0.0
    hits = 0
    for _ in range(trials):
        for g in gold:
            picks = rng.sample(index.corpus, min(top_k, len(index.corpus)))
            hits += any(matches(g, c["text"]) for c in picks)
    return hits / (trials * max(1, len(gold)))


def evaluate(index, gold: List[Dict], answer_fn: Optional[Callable] = None,
             show: int = 5, quiet: bool = False) -> Dict:
    """MRR / hit@1 / hit@3 / hit@k, plus end-to-end accuracy if answer_fn given."""
    rr, h1, h3, hk, acc = [], 0, 0, 0, 0
    fails = []

    for g in gold:
        chunks = index.retrieve(g["q"])
        rank = next((i + 1 for i, c in enumerate(chunks) if matches(g, c["text"])), 0)
        rr.append(1.0 / rank if rank else 0.0)
        h1 += rank == 1
        h3 += 0 < rank <= 3
        hk += rank > 0

        ans, ok = "", None
        if answer_fn:
            ans = answer_fn(g["q"])
            ok = matches(g, ans)
            acc += bool(ok)
        if rank == 0 or (answer_fn and not ok):
            fails.append((rank, ok, g, ans))

    n = max(1, len(gold))
    m = {"n": len(gold), "mrr": float(np.mean(rr)) if rr else 0.0,
         "hit1": h1 / n, "hit3": h3 / n, "hitk": hk / n,
         "acc": acc / n if answer_fn else None}

    if not quiet:
        line = (f"MRR {m['mrr']:.3f} | hit@1 {m['hit1']:.0%} | "
                f"hit@3 {m['hit3']:.0%} | hit@k {m['hitk']:.0%}")
        if answer_fn:
            line += f" | acc {acc}/{len(gold)} ({m['acc']:.0%})"
        print(line)
        for rank, ok, g, ans in fails[:show]:
            tag = f"rank={rank or 'MISS'}" + (f" {'OK' if ok else 'BAD'}" if answer_fn else "")
            print(f"  {tag:<16}| {g['q'][:58]}")
            print(f"      want={str(g.get('answer'))[:46]!r}")
            if answer_fn:
                print(f"      got ={ans[:66]!r}")
    return m


DEFAULT_VARIANTS = {
    "bm25 only":        {"use_vector": False, "use_bm25": True,  "use_rerank": False},
    "vector only":      {"use_vector": True,  "use_bm25": False, "use_rerank": False},
    "hybrid (RRF)":     {"use_vector": True,  "use_bm25": True,  "use_rerank": False},
    "hybrid, no ngram": {"use_vector": True,  "use_bm25": True,  "use_rerank": False, "ngram": 0},
    "hybrid + rerank":  {"use_vector": True,  "use_bm25": True,  "use_rerank": True},
}


def ablate(index, gold: List[Dict], variants: Optional[Dict] = None) -> List:
    """One variable at a time. Retrieval-only, so it costs no API calls."""
    variants = variants or DEFAULT_VARIANTS
    base = copy.deepcopy(index.cfg)
    rows = []
    for label, ov in variants.items():
        index.cfg.update(base)
        index.cfg.update(ov)
        if index.cfg["ngram"] != base["ngram"]:
            index.rebuild_bm25()
        m = evaluate(index, gold, quiet=True)
        rows.append((label, m))
    index.cfg.update(base)
    index.rebuild_bm25()

    print(f"{'variant':<22}{'MRR':>8}{'hit@1':>8}{'hit@3':>8}{'hit@k':>8}")
    for label, m in rows:
        print(f"{label:<22}{m['mrr']:>8.3f}{m['hit1']:>8.0%}{m['hit3']:>8.0%}{m['hitk']:>8.0%}")
    return rows
