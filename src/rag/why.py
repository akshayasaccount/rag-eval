# src/rag/why.py - error analysis for one gold question: what each mode returns, which hits are gold
#   uv run src/rag/why.py q001
import json, re, sys
from config import CFG
from retrieve import Retriever
from rerank import Reranker

def norm(s): return re.sub(r"\s+", " ", s.lower()).strip()

qid = sys.argv[1]
g = next(x for x in (json.loads(l) for l in open("eval/gold.jsonl", encoding="utf-8") if l.strip()) if x["qid"] == qid)
r, rr = Retriever(), Reranker(CFG["rerank_model"])
spans = g["evidence"] if isinstance(g["evidence"], list) else [g["evidence"]]
gold = {c["chunk_id"] for c in r.chunks
        if c["doc_id"] == g["doc_id"] and any(norm(s) in norm(c["text"]) for s in spans)}
print("Q:", g["question"]); print("gold:", sorted(gold))
runs = {m: r.search(g["question"], m, 10) for m in ("dense", "bm25", "hybrid")}
runs["hybrid+rerank"] = rr.rerank(g["question"], r.candidates(g["question"]), 10)
for mode, hits in runs.items():
    print(f"\n{mode}")
    for i, h in enumerate(hits, 1):
        mark = "GOLD" if h["chunk_id"] in gold else "    "
        print(f"  {i:2} {mark} {h['chunk_id']:18} p{h['page_start']:<3} {h['text'][:90]}")