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

    ## 2026-10-25 - Step 4: BM25 + hybrid retrieval
- retrieve.py: Retriever with dense / bm25 / hybrid (RRF k=60, pool=30) behind one search() interface
- Same 3 smoke questions, top-5:
  - RAGAS question: dense 0/5 from 2309.15217v2; bm25 rank 3 (:15 p5); hybrid rank 4 (:0 p1)
    -> hybrid rescues the proper-noun query. BUT rank 1 in bm25 AND hybrid is the survey's
    bibliography chunk 2601.05264v1:80 (p73): reference lists are keyword-dense and win BM25.
    Strongest evidence yet for the Step 8 reference filter.
  - R2AG question: all three modes 5/5 from 2406.13249v2
  - dense-vs-sparse: hybrid mixes survey + reproducibility-study chunks; plausible, unverifiable yet
- Open: the chunk defining RAGAS's metrics is not top-5 in any mode. Step 5 reranker + Step 7
  gold set will show whether it is a ranking problem or a recall problem.

  ## 2026-10-25 - Step 5: cross-encoder reranking
- rerank.py: cross-encoder/ms-marco-MiniLM-L-6-v2 over the hybrid top-30 pool, keep top-5
- RAGAS question: bibliography chunk 2601.05264v1:80 dropped out of top-5 (hypothesis confirmed),
  but new #1 is 2601.05264v1:54 (governance maturity table mentioning "evaluation") - reranker false positive.
  Only 2 RAGAS-paper chunks in the 30-pool (:0 abstract -> rank 5, :15 -> rank 14); the chunk that
  defines the three metrics is NOT in the pool. => RECALL problem, not a ranking problem.
- R2AG: 5/5 after rerank. dense-vs-sparse: reordered, unverifiable yet.
- Step 8 candidates for recall: title injection (prepend paper title to every chunk before embed/BM25);
  bge query-instruction prefix; pool 30 -> 100; bge-base embedder; bge-reranker-base; references filter.
- 3 smoke questions are anecdotes. Decisions wait for the 80-question gold set (Step 7).

## 2026-10-26 - Step 6: generation
- generate.py: OpenAI-compatible client (Ollama default, hosted via 3 env vars); Groq free tier, model ___;
  temperature 0, max_tokens 1000
- Prompt: context chunks tagged [chunk_id] (title, page) inside <context> tags; answer only from context;
  cite ids; exact refusal string
- R2AG question: grounded answer citing :4, :5, :35 (all in the 5 given). First run truncated at
  max_tokens=400 -> raised to 1000.
- Refusal test ("capital of France"): "Not found in the provided papers.", no citations. First run echoed
  a Q/A pattern from a context chunk -> wrapped context in tags, dropped trailing "Answer:" cue.
- Ollama (local) path kept as the no-key default; switched to Groq for speed.

## 2026-10-26 - Step 6: generation
- generate.py: OpenAI-compatible client (Ollama default, hosted via 3 env vars); Groq free tier, model ___;
  temperature 0, max_tokens 1000
- Prompt: context chunks tagged [chunk_id] (title, page) inside <context> tags; answer only from context;
  cite ids; exact refusal string
- R2AG question: grounded answer citing :0, :4, :35 (all in the 5 given). First run truncated at
  max_tokens=400 (reasoning tokens count) -> raised to 1000.
- Refusal test ("capital of France"): "Not found in the provided papers.", no citations. First run echoed
  a Q/A pattern from a context chunk -> wrapped context in tags, dropped trailing "Answer:" cue.
- Ollama (local) path kept as the no-key default; switched to Groq for speed.

## 2026-10-26 - Step 7: gold set + baseline eval (n=21)
- Gold: 21 questions, 5 papers, evidence = verbatim span (robust to re-chunking); LLM-drafted, human-reviewed
- Baseline retrieval (recall@1/5/10, MRR@10):
  dense 0.29/0.52/0.62/0.40 (8 misses) | bm25 0.38/0.81/0.95/0.59 (miss: q002, the paraphrase)
  hybrid 0.38/0.81/0.90/0.54 (q001,q018) | hybrid+rerank 0.62/0.90/0.90/0.74 (q001,q006)
- Findings: BM25 >> dense here (questions name papers/methods; gold set is lexically biased, only 4 paraphrases);
  hybrid slightly below bm25 (RRF dilution); reranker lifts r@1 by 24 pts and MRR by 20 pts but cannot fix recall;
  q001 (RAGAS metrics) missed by all but bm25 -> reranker pool is fused top-30, not the union -> bug/knob
- Caveat: n=21, 1 question = 4.8 pts; nothing under ~10 pts is a real difference
- Step 8 order: (1) rerank over union of dense+bm25 lists, (2) title injection, (3) bge query instruction,
  (4) BM25-weighted RRF, (5) bge-base embedder, (6) bigger reranker, (7) references filter, (8) chunk size
- Gold set: grow to 80 with >= 15 paraphrases and 10 not-in-corpus
## 2026-10-26 - Step 7: gold set + baseline eval (n=21)
- Gold: 21 questions, 5 papers, evidence = verbatim span (robust to re-chunking); LLM-drafted, human-reviewed
- Baseline retrieval (recall@1/5/10, MRR@10):
  dense 0.29/0.52/0.62/0.40 (8 misses) | bm25 0.38/0.81/0.95/0.59 (miss: q002, the paraphrase)
  hybrid 0.38/0.81/0.90/0.54 (q001,q018) | hybrid+rerank 0.62/0.90/0.90/0.74 (q001,q006)
- Findings: BM25 >> dense here (questions name papers/methods; gold set is lexically biased, only 4 paraphrases);
  hybrid slightly below bm25 (RRF dilution); reranker lifts r@1 by 24 pts and MRR by 20 pts but cannot fix recall;
  q001 (RAGAS metrics) missed by all but bm25 -> reranker pool is fused top-30, not the union -> bug/knob
- Caveat: n=21, 1 question = 4.8 pts; nothing under ~10 pts is a real difference
- Step 8 order: (1) rerank over union of dense+bm25 lists, (2) title injection, (3) bge query instruction,
  (4) BM25-weighted RRF, (5) bge-base embedder, (6) bigger reranker, (7) references filter, (8) chunk size
- Gold set: grow to 80 with >= 15 paraphrases and 10 not-in-corpus

## 2026-10-26 - Step 8 runs 0-1
- baseline_v2 (refactored config-driven code): identical to baseline -> regression check passed
- pool_union=1: hybrid+rerank unchanged (0.62/0.90/0.90/0.73), q001 + q006 still missed, eval time 58s -> 118s.
  Gold chunks ARE in the pool; MiniLM-L6 reranker demotes them. Ranking problem now, not recall. Reverted.
- by type (r@5): dense factual 0.22 vs bm25 0.89 -> dense fails on paper/method names; hybrid paraphrase 0.75 > bm25 0.50
- next: swap reranker (bge-reranker-base), then title injection for dense
- reranker_bge_base: hybrid+rerank 0.52/0.86/0.95/0.67, miss q001 only (q006 recovered) but r@1 -10 pts, MRR -6 pts,
  350s vs 58s. Within noise at n=21, 6x slower -> not adopted. q001 still missed with its chunks in the pool.
- Step 8 paused after 3 runs (baseline, pool_union, reranker_bge_base). title_inject deferred until gold set = 80.
