# src/rag/eval.py - Step 7/8: retrieval metrics on the gold set, logged to results.jsonl + MLflow
#   uv run src/rag/eval.py                 print the table
#   uv run src/rag/eval.py --tag <name>    also log to eval/results.jsonl and MLflow experiment "rag-eval"
import json, re, sys, time
from collections import defaultdict
from config import CFG
from retrieve import Retriever
from rerank import Reranker

GOLD = "eval/gold.jsonl"
RESULTS = "eval/results.jsonl"
KS = (1, 5, 10)

def norm(s): return re.sub(r"\s+", " ", s.lower()).strip()

def gold_chunk_ids(g, chunks):
    return {c["chunk_id"] for c in chunks
            if c["doc_id"] == g["doc_id"] and norm(g["evidence"]) in norm(c["text"])}

def score(ranked_ids, gold_ids):
    hit = next((i for i, cid in enumerate(ranked_ids, 1) if cid in gold_ids), None)
    return {f"recall@{k}": float(hit is not None and hit <= k) for k in KS}, (1 / hit if hit else 0.0)

def main():
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else None
    gold = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    gold = [g for g in gold if g["type"] != "not-in-corpus"]
    r, rr = Retriever(), Reranker(CFG["rerank_model"])
    modes = {
        "dense":         lambda q: [h["chunk_id"] for h in r.search(q, "dense", 10)],
        "bm25":          lambda q: [h["chunk_id"] for h in r.search(q, "bm25", 10)],
        "hybrid":        lambda q: [h["chunk_id"] for h in r.search(q, "hybrid", 10)],
        "hybrid+rerank": lambda q: [h["chunk_id"] for h in rr.rerank(q, r.candidates(q), 10)],
    }
    print(f"{len(gold)} questions   config: {CFG}\n")
    print(f'{"mode":15}' + "".join(f"{f'recall@{k}':>10}" for k in KS) + f'{"MRR@10":>10}')
    rows = {}
    for name, fn in modes.items():
        totals = {f"recall@{k}": 0.0 for k in KS}; mrr = 0.0; misses = []
        by_type = defaultdict(lambda: [0.0, 0]); t0 = time.time()
        for g in gold:
            rec, rrank = score(fn(g["question"]), gold_chunk_ids(g, r.chunks))
            for k in totals: totals[k] += rec[k]
            mrr += rrank
            by_type[g["type"]][0] += rec["recall@5"]; by_type[g["type"]][1] += 1
            if rec["recall@10"] == 0: misses.append(g["qid"])
        n = len(gold)
        row = {k: v / n for k, v in totals.items()} | {"MRR@10": mrr / n}
        rows[name] = (row, misses)
        print(f"{name:15}" + "".join(f"{row[f'recall@{k}']:>10.2f}" for k in KS)
              + f"{row['MRR@10']:>10.2f}   ({time.time() - t0:.0f}s)")
        print("   r@5 by type: " + "  ".join(f"{t} {s / c:.2f} ({c})" for t, (s, c) in sorted(by_type.items())))
        print("   miss@10:", misses or "none")
    if tag:
        with open(RESULTS, "a", encoding="utf-8") as f:
            for name, (row, misses) in rows.items():
                f.write(json.dumps({"tag": tag, "mode": name, "n": len(gold), **row,
                                    "misses": misses, "cfg": CFG}) + "\n")
        import mlflow
        mlflow.set_experiment("rag-eval")
        with mlflow.start_run(run_name=tag):
            mlflow.log_params({**CFG, "n_questions": len(gold)})
            for name, (row, _) in rows.items():
                safe = name.replace("+", "_")
                mlflow.log_metrics({f"{safe}/{k.replace('@', '_at_')}": v for k, v in row.items()})
        print(f"\nlogged '{tag}' to {RESULTS} and MLflow (view: uv run mlflow ui)")

if __name__ == "__main__":
    main()