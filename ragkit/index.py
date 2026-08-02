from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from typing import Dict, Iterable, List, Optional

import numpy as np

DEFAULTS = {
    "embed_model": "BAAI/bge-m3",
    "query_prefix": "",          # multilingual-e5 needs "query: "
    "doc_prefix": "",            # multilingual-e5 needs "passage: "
    "chunk_size": 700,
    "chunk_overlap": 120,
    "ngram": 4,                  # 0 disables character n-grams
    "use_vector": True,
    "use_bm25": True,
    "use_rerank": True,
    "candidates": 60,
    "rrf_k": 60,
    "rerank_pool": 40,
    "top_k": 8,
    "dedupe": 0.85,
    "rerank_model": "BAAI/bge-reranker-v2-m3",
}

_WORD_RE = re.compile(r"[\u0980-\u09FF\u0900-\u097F\u0600-\u06FF]+|[a-z0-9]+")


def tokenize(text: str, ngram: int = 4) -> List[str]:
    text = unicodedata.normalize("NFC", text or "").lower()
    words = _WORD_RE.findall(text)
    if not ngram:
        return words
    grams: List[str] = []
    for w in words:
        grams.append(w)
        if len(w) > ngram:
            grams.extend(w[i:i + ngram] for i in range(len(w) - ngram + 1))
    return grams


def overlap(a: str, b: str) -> float:
    ta, tb = set(_WORD_RE.findall(a.lower())), set(_WORD_RE.findall(b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# --------------------------------------------------------------------------
def chunk_document(pages: Dict[int, str], doc_id: str,
                   chunk_size: int = 700, chunk_overlap: int = 120,
                   min_chars: int = 40) -> List[Dict]:
    """Split one document's pages into retrievable chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        # `।` (U+0964) sits in the Devanagari block, not Bengali — a naive
        # [^\u0980-\u09FF] filter upstream deletes every sentence boundary.
        separators=["\n\n", "\n", "। ", "।", "۔ ", "? ", "! ", ". ", " ", ""],
        length_function=len, keep_separator=True)

    out = []
    for page_no in sorted(pages):
        text = (pages[page_no] or "").strip()
        if len(text) < min_chars:
            continue
        for j, piece in enumerate(splitter.split_text(text)):
            piece = piece.strip()
            if len(piece) < min_chars:
                continue
            out.append({
                "id": f"{doc_id}::p{page_no:04d}_c{j:03d}",
                "text": piece,
                "doc": doc_id,
                "page": page_no,
            })
    return out


# --------------------------------------------------------------------------
class HybridIndex:
    """Vector + BM25 over one Chroma collection holding many documents."""

    def __init__(self, db_path: str, collection: str = "corpus",
                 cfg: Optional[Dict] = None, emb_cache: Optional[str] = None):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.emb_cache = emb_cache
        os.makedirs(db_path, exist_ok=True)
        if emb_cache:
            os.makedirs(emb_cache, exist_ok=True)

        self.model = SentenceTransformer(self.cfg["embed_model"])
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"})
        self._reranker = None
        self.corpus: List[Dict] = []
        self.bm25 = None
        self.reload()

    # ---------- embedding ----------
    def embed(self, texts: List[str], prefix: str = "", cache_key: str = "") -> List[List[float]]:
        path = None
        if cache_key and self.emb_cache:
            h = hashlib.md5(f"{self.cfg['embed_model']}|{cache_key}|{len(texts)}"
                            .encode()).hexdigest()[:16]
            path = os.path.join(self.emb_cache, f"{h}.npy")
            if os.path.exists(path):
                return np.load(path).tolist()
        vecs = np.asarray(self.model.encode(
            [prefix + t for t in texts], batch_size=16,
            show_progress_bar=len(texts) > 64, normalize_embeddings=True))
        if path:
            np.save(path, vecs)
        return vecs.tolist()

    # ---------- indexing ----------
    def reload(self):
        got = self.collection.get(include=["documents", "metadatas"])
        if not got["ids"]:
            self.corpus, self.bm25 = [], None
            return
        self.corpus = [
            {"id": i, "text": d, **{k: (m or {}).get(k) for k in ("doc", "page", "section")}}
            for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
        ]
        self.rebuild_bm25()

    def rebuild_bm25(self):
        """Cheap. Call after changing cfg['ngram'] — no re-embedding needed."""
        from rank_bm25 import BM25Okapi
        n = self.cfg["ngram"]
        self.bm25 = BM25Okapi([tokenize(c["text"], n) for c in self.corpus])

    def add(self, chunks: List[Dict], cache_key: str = ""):
        if not chunks:
            return
        existing = set(self.collection.get(include=[])["ids"])
        new = [c for c in chunks if c["id"] not in existing]
        if not new:
            print(f"  all {len(chunks)} chunks already indexed")
            self.reload()
            return
        embs = self.embed([c["text"] for c in new], self.cfg["doc_prefix"], cache_key)
        for i in range(0, len(new), 256):
            sl = slice(i, i + 256)
            self.collection.add(
                ids=[c["id"] for c in new[sl]],
                embeddings=embs[sl],
                documents=[c["text"] for c in new[sl]],
                metadatas=[{"doc": c["doc"], "page": c["page"],
                            "section": c.get("section", "prose")} for c in new[sl]])
        self.reload()

    @property
    def docs(self) -> List[str]:
        return sorted({c["doc"] for c in self.corpus})

    # ---------- retrieval ----------
    def _rerank(self, query: str, pool: List[Dict]) -> List[Dict]:
        if self._reranker is False:               # tried once, unavailable
            return pool
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                try:
                    import torch
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    dev = "cpu"
                self._reranker = CrossEncoder(self.cfg["rerank_model"],
                                              max_length=512, device=dev)
            except Exception as e:
                print(f"  reranker unavailable ({e}); continuing without it")
                self._reranker = False
                return pool
        scores = self._reranker.predict([(query, c["text"]) for c in pool])
        for c, s in zip(pool, scores):
            c["rerank"] = float(s)
        return sorted(pool, key=lambda c: c["rerank"], reverse=True)

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 docs: Optional[Iterable[str]] = None,
                 sections: Optional[Iterable[str]] = None,
                 debug: bool = False) -> List[Dict]:
        if not self.corpus:
            return []
        cfg = self.cfg
        top_k = top_k or cfg["top_k"]
        where = {"doc": {"$in": list(docs)}} if docs else None

        ranks: Dict[str, float] = {}
        k, cand = cfg["rrf_k"], min(cfg["candidates"], len(self.corpus))

        if cfg["use_vector"]:
            qv = self.embed([query], cfg["query_prefix"])[0]
            res = self.collection.query(query_embeddings=[qv], n_results=cand,
                                        where=where)
            for r, cid in enumerate(res["ids"][0]):
                ranks[cid] = ranks.get(cid, 0.0) + 1.0 / (k + r + 1)

        if cfg["use_bm25"] and self.bm25 is not None:
            sc = self.bm25.get_scores(tokenize(query, cfg["ngram"]))
            allowed = set(docs) if docs else None
            for r, i in enumerate(np.argsort(sc)[::-1][:cand * 2]):
                if sc[i] <= 0:
                    continue
                c = self.corpus[i]
                if allowed and c["doc"] not in allowed:
                    continue
                ranks[c["id"]] = ranks.get(c["id"], 0.0) + 1.0 / (k + r + 1)

        by_id = {c["id"]: c for c in self.corpus}
        pool = [{**by_id[cid], "rrf": s} for cid, s in
                sorted(ranks.items(), key=lambda kv: kv[1], reverse=True) if cid in by_id]

        if sections:
            keep = set(sections)
            pool = [c for c in pool if c.get("section") in keep]

        if cfg["use_rerank"] and len(pool) > 1:
            pool = self._rerank(query, pool[:cfg["rerank_pool"]])

        selected: List[Dict] = []
        for c in pool:
            if len(selected) >= top_k:
                break
            if any(overlap(c["text"], s["text"]) > cfg["dedupe"] for s in selected):
                continue
            selected.append(c)

        if debug:
            print(f"\n[retrieve] {query!r}")
            for c in selected:
                rr = f" rr={c['rerank']:+.3f}" if "rerank" in c else ""
                print(f"  {c['doc'][:18]:<18} p{c['page']:>3} [{str(c.get('section'))[:9]:<9}]"
                      f" rrf={c['rrf']:.4f}{rr} | {c['text'][:70]}...")
        return selected
