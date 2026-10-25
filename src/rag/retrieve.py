# src/rag/retrieve.py - Step 4: dense, BM25 and hybrid retrieval over chunks.jsonl
import json, re
import faiss, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS = "data/processed/chunks.jsonl"
INDEX = "data/processed/index.faiss"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOKEN = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())

class Retriever:
    def __init__(self):
        self.chunks = [json.loads(l) for l in open(CHUNKS)]
        self.index = faiss.read_index(INDEX)
        self.model = SentenceTransformer(MODEL_NAME)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def dense(self, q: str, k: int = 10) -> list[tuple[int, float]]:
        qv = self.model.encode([q], normalize_embeddings=True).astype("float32")
        scores, idx = self.index.search(qv, k)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0])]

    def bm25_search(self, q: str, k: int = 10) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(q))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]

    def hybrid(self, q: str, k: int = 10, pool: int = 30, rrf_k: int = 60) -> list[tuple[int, float]]:
        """Reciprocal rank fusion of the dense and BM25 rankings."""
        fused: dict[int, float] = {}
        for ranking in (self.dense(q, pool), self.bm25_search(q, pool)):
            for rank, (i, _) in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]

    def search(self, q: str, mode: str = "hybrid", k: int = 10) -> list[dict]:
        fn = {"dense": self.dense, "bm25": self.bm25_search, "hybrid": self.hybrid}[mode]
        return [{**self.chunks[i], "score": s, "rank": r}
                for r, (i, s) in enumerate(fn(q, k), start=1)]