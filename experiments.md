## 2026-10-24 - Step 1: corpus + ingestion
- Corpus: arXiv, query all:"retrieval augmented generation", top 50 by relevance
- Extractor: PyMuPDF page.get_text() + hyphen/whitespace cleanup; arxiv.py 4.0.1 (pdf_url + urlretrieve)
- Stats: 50 papers, 883 pages, docs.jsonl 3.39 MB
- Spot-check: 3/3 samples read in order; two-column layout OK
- Known issues: hard line breaks mid-sentence from PDF layout; one 86-page survey (2601.05264v1)
  will dominate chunk count; References sections still included
- Deferred to Step 2: join single newlines, strip References (ablate), cap pages per doc?

## 2026-10-25 - Step 2: chunking
- Settings: CHUNK_WORDS=250, OVERLAP_WORDS=50, STRIP_REFERENCES=False
- Tried STRIP_REFERENCES=True (cut at References heading, 2nd-half guard). check_strip.py:
  13/50 papers cut at 52-71% (appendices after references lost, e.g. 2309.15217v2 8p -> 5p),
  15/50 untouched (heading format not matched). Inconsistent -> disabled.
- Result: 2319 chunks from 50 docs; chunks/doc median __ 29.5_, max _ 157__ (___); words/chunk median 250
- Line-join fix applied; samples read clean; chunk 0 of each paper is title/authors/affiliations
- Step 8 ablations: chunk-level reference filter by citation density (keeps appendices),
  chunk size 120/250/400, neighbour-chunk expansion

  ## 2026-10-25 - Step 3: embeddings + index
- Model: BAAI/bge-small-en-v1.5 (384-dim, normalized), FAISS IndexFlatIP (exact), 2319 vectors
- Smoke test (dense only, top-5):
  - R2AG question: 5/5 hits from 2406.13249v2 (0.82-0.89)
  - dense-vs-sparse question: sensible (survey definitions, reproducibility study)
  - RAGAS question: MISS - 0/5 from 2309.15217v2; returned generic evaluation chunks
    and one bibliography chunk (2601.05264v1:80). Proper noun blurred by embeddings.
    -> motivates BM25 + hybrid (Step 4) and the references filter (Step 8)