# src/rag/check_retrieve.py - compare dense / bm25 / hybrid on the same questions
from retrieve import Retriever

r = Retriever()
QUESTIONS = [
    "What metrics does RAGAS use to evaluate a RAG pipeline?",
    "How does R2AG incorporate retrieval information into generation?",
    "What is the difference between dense and sparse retrieval?",
]
for q in QUESTIONS:
    print("\nQ:", q)
    for mode in ("dense", "bm25", "hybrid"):
        hits = r.search(q, mode=mode, k=5)
        print(f"  {mode:7}", "  ".join(f'{h["chunk_id"]}(p{h["page_start"]})' for h in hits))