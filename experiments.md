## 2026-10-24 - Step 1: corpus + ingestion
- Corpus: arXiv, query all:"retrieval augmented generation", top 50 by relevance
- Extractor: PyMuPDF page.get_text() + hyphen/whitespace cleanup; arxiv.py 4.0.1 (pdf_url + urlretrieve)
- Stats: 50 papers, 883 pages, docs.jsonl 3.39 MB
- Spot-check: 3/3 samples read in order; two-column layout OK
- Known issues: hard line breaks mid-sentence from PDF layout; one 86-page survey (2601.05264v1)
  will dominate chunk count; References sections still included
- Deferred to Step 2: join single newlines, strip References (ablate), cap pages per doc?
