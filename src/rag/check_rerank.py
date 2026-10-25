# src/rag/check_rerank.py - does the cross-encoder fix the ordering?
from retrieve import Retriever
from rerank import Reranker

r, rr = Retriever(), Reranker()
QUESTIONS = [
    "What metrics does RAGAS use to evaluate a RAG pipeline?",
    "How does R2AG incorporate retrieval information into generation?",
    "What is the difference between dense and sparse retrieval?",
]
def show(label, hits):
    print(f"  {label:14}", "  ".join(f'{h["chunk_id"]}(p{h["page_start"]})' for h in hits))

for q in QUESTIONS:
    print("\nQ:", q)
    pool = r.search(q, mode="hybrid", k=30)
    show("hybrid top5", pool[:5])
    show("reranked top5", rr.rerank(q, pool, k=5))

# the open question from Step 4: is the RAGAS paper even in the pool?
q = QUESTIONS[0]
pool = r.search(q, mode="hybrid", k=30)
in_pool = [h["chunk_id"] for h in pool if h["doc_id"] == "2309.15217v2"]
print("\nRAGAS-paper chunks in the 30-pool:", in_pool or "none")
top = rr.rerank(q, pool, k=30)
print("after rerank:", [(h["chunk_id"], h["rerank_rank"]) for h in top if h["doc_id"] == "2309.15217v2"] or "none")
print("\nTop reranked chunk:", top[0]["chunk_id"])
print(top[0]["text"][:400])