# src/rag/rerank.py - Step 5: cross-encoder reranking of a retrieved pool
from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # fast on CPU; try BAAI/bge-reranker-base in Step 8

class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, hits: list[dict], k: int = 5) -> list[dict]:
        """Score every (question, chunk) pair jointly; return the top k by that score."""
        scores = self.model.predict([(question, h["text"]) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:k]
        return [{**h, "rerank_score": float(s), "rerank_rank": r}
                for r, (h, s) in enumerate(ranked, start=1)]