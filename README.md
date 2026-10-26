# rag-eval: measured RAG over 50 arXiv papers

Question answering over 50 arXiv papers on retrieval-augmented generation, built to be measured rather than demoed:
every retrieval choice is scored on a hand-verified gold set, and every decision is logged in `experiments.md`.

**Live demo:** https://akshayareddy024--rag-eval-web.us-east.modal.run (first request after idle takes ~30 s)

## What it does

- Ingests 50 papers from the arXiv API (883 pages) and cuts them into 2,319 passages with page spans
- Retrieves with dense (bge-small + FAISS), BM25, or hybrid (reciprocal rank fusion), then reranks with a cross-encoder
- Answers with an LLM that may only use the retrieved passages, cites chunk ids, and refuses when the answer is absent
- Scores every retrieval mode on a gold set (recall@k, MRR) and logs each run to MLflow

## Results (gold set n=21, retrieval)

| mode | recall@1 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|
| dense (bge-small) | 0.29 | 0.52 | 0.62 | 0.40 |
| BM25 | 0.38 | 0.81 | 0.95 | 0.59 |
| hybrid (RRF) | 0.38 | 0.81 | 0.90 | 0.54 |
| hybrid + rerank (MiniLM-L6) | 0.62 | 0.90 | 0.90 | 0.74 |
| hybrid + rerank (bge-reranker-base) | 0.52 | 0.86 | 0.95 | 0.67 |

With 21 questions one question is 4.8 points, so differences under ~10 points are noise. The gold set is being grown to 80.

## What the numbers taught me

1. **Keyword search beats embeddings on this corpus.** Questions name papers and methods ("R2AG", "MetaRAG"); BM25 keys on those tokens, bge-small blurs them. BM25's single miss is the one true paraphrase question, which only dense retrieval finds.
2. **Hybrid scores below BM25 alone** at this sample size: fusing a strong ranker with a weak one dilutes it.
3. **Reranking is the biggest win for ordering**: recall@1 +24 pts, MRR +20 pts. It cannot fix recall; a chunk the retrievers never fetch cannot be promoted.
4. **Bibliography chunks win keyword search.** Reference lists are keyword-dense, so a survey's bibliography outranked real content for a question that named a paper. Motivates a reference filter (next).
5. **A bigger reranker was a trade, not a win**: +1 question at recall@10, -2 at recall@1, 6x slower. Not adopted at n=21.
6. **Error analysis beats guessing.** `why.py <qid>` prints what every mode returns for one question with gold hits marked. It showed the one stubborn miss was neither a label gap nor a recall gap, but a reranker demotion.

## Evaluation design

- Gold evidence is a verbatim text span, not a chunk id, so the gold set survives re-chunking and chunk-size ablations stay valid
- Questions were LLM-drafted from passages, then human-reviewed; about half were rejected (citation trivia, table numbers, ambiguous table rows)
- Metrics: recall@{1,5,10} and MRR@10 per retrieval mode, broken down by question type (factual / paraphrase / numeric)
- Known limitation: only the source paper's chunks count as gold, so cross-paper restatements are not credited and recall is a lower bound

## Architecture

```
arXiv API -> PDFs -> PyMuPDF page text -> 250-word chunks (50 overlap, page spans)
  -> bge-small embeddings + FAISS (exact cosine)  -> dense top-30  \
  -> BM25 over the same chunks                     -> bm25 top-30   / RRF -> pool
  -> cross-encoder rerank -> top-5 -> LLM (OpenAI-compatible: Groq or local Ollama) -> cited answer | refusal
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run src/rag/ingest.py      # 50 papers from arXiv -> data/processed/docs.jsonl
uv run src/rag/chunk.py       # -> chunks.jsonl
uv run src/rag/embed.py       # -> index.faiss
uv run src/rag/eval.py        # retrieval metrics on eval/gold.jsonl
```

Answering needs an LLM: copy `.env.example` to `.env` and fill in any OpenAI-compatible endpoint, then

```
uv run uvicorn app:app --app-dir src/rag --port 7860    # UI at http://127.0.0.1:7860, API docs at /docs
```

## Deploy

`Dockerfile` builds a CPU-only image with both models baked in (port 7860). `modal_app.py` deploys the same service on
Modal and scales to zero: create a `rag-eval-llm` secret with the three LLM variables, then `uv run modal deploy modal_app.py`.

## Project layout

```
src/rag/
  ingest.py     arXiv -> page text            retrieve.py   dense / BM25 / hybrid (config-driven)
  chunk.py      page text -> chunks           rerank.py     cross-encoder reranking
  embed.py      chunks -> FAISS index         generate.py   cited answers via OpenAI-compatible API
  config.py     every knob, RAG_* env vars    app.py        FastAPI service + UI
  eval.py       gold-set metrics + MLflow     why.py        per-question error analysis
  gold_tool.py / gold_draft.py / gold_review.py   gold-set tooling
eval/gold.jsonl      hand-verified questions     eval/results.jsonl   every logged run
experiments.md       the lab notebook: every decision, number and dead end
```

## Next

- Grow the gold set to 80 (15+ paraphrases, 10 not-in-corpus) so differences become significant
- Title injection (prepend the paper title to each chunk) to fix dense retrieval on paper names
- Chunk-level reference filter by citation density
- Chunk size 120 / 250 / 400 ablation
- Answer-level metrics: faithfulness and correctness with an LLM judge, abstention accuracy on not-in-corpus questions

## Stack

Python, uv, PyMuPDF, sentence-transformers, FAISS, rank-bm25, FastAPI, MLflow, Docker, Modal, Groq (LLM)
