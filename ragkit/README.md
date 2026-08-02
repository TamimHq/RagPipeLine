# ragkit — document-agnostic RAG

Hybrid retrieval over a folder of PDFs. Built and measured on two very
different corpora: an English born-digital CV and a Bengali exam textbook whose
PDF text layer is broken.

## Why the defaults are what they are

Each of these was measured, not assumed.

| Choice | Evidence |
|---|---|
| Per-page OCR decision via script coherence | Font metadata gave false positives (base-14 fonts have no ToUnicode legitimately). Malformed-mark ratio: 0.00 valid Bengali vs 0.32-0.40 mojibake |
| Character n-gram BM25 | `ভাগ্যদেবতার` shares zero whole-word tokens with a query saying `ভাগ্য দেবতা` |
| Reciprocal Rank Fusion | Cosine is bounded [0,1], BM25 is not; weighted sums are dominated by BM25 |
| Cosine + normalized vectors | Chroma defaults to L2, making `1 - distance` meaningless |
| **No section quota** | Reserving slots for "answer-bearing" sections measured MRR 0.29 vs 0.93 without it |
| Rank-sensitive metrics | hit@k read 96-100% across variants whose MRR ranged 0.22-0.93 |
| Gold filtering by target count | Answers appearing 100+ times get "hit" by luck and inflate every score |

## Use

```python
from index import HybridIndex
from pipeline import build_corpus, RAG

index = HybridIndex(db_path="./chroma_v1", emb_cache="./emb")
build_corpus("./pdfs", index, ocr_fn=my_ocr)      # ocr_fn optional
rag = RAG(index, generate=my_llm)
rag.ask("...")                                     # or docs=["report_a1b2"]
```

## Evaluate

```python
from evaluate import gold_from_answer_keys, gold_from_llm, filter_gold, evaluate, ablate, random_baseline

gold = gold_from_answer_keys(chunks) or gold_from_llm(chunks, generate=my_llm)
gold = filter_gold(gold, index.corpus, max_targets=3)
print("chance:", random_baseline(index, gold))
evaluate(index, gold, answer_fn=rag.ask)
ablate(index, gold)
```

## Layout

- `extract.py` — text-layer quality detection, boilerplate learning
- `sections.py` — content-based chunk classification
- `index.py` — chunking, tokenization, hybrid retrieval
- `pipeline.py` — folder ingestion, RAG answerer
- `evaluate.py` — gold generation, metrics, ablation
