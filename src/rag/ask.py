# src/rag/ask.py - end-to-end: question -> hybrid -> rerank -> LLM answer with citations
import sys
from retrieve import Retriever
from rerank import Reranker
from generate import Generator

def main():
    question = " ".join(sys.argv[1:]) or "What metrics does RAGAS use to evaluate a RAG pipeline?"
    r, rr, g = Retriever(), Reranker(), Generator()
    pool = r.search(question, mode="hybrid", k=30)
    top = rr.rerank(question, pool, k=5)
    out = g.answer(question, top)
    print("\nQ:", question)
    print("\nA:", out["answer"])
    print("\nCited:", out["cited"])
    print("Context given:", out["context_ids"])

if __name__ == "__main__":
    main()