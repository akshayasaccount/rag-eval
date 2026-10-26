# src/rag/retrieve.py - Step 4: dense, BM25 and hybrid retrieval over chunks.jsonl
import json, re
import faiss, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
# src/rag/retrieve.py - Step 4/8: dense, BM25 and hybrid retrieval, config-aware
import json, re
from pathlib import Path
import faiss, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from config import CFG, index_text, query_text

CHUNKS = "data/processed/chunks.jsonl"
INDEX = "data/processed/index.faiss"
META = Path("data/processed/index_meta.json")
TOKEN = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())

class Retriever:
    def __init__(self):
        meta = json.loads(META.read_text()) if META.exists() else {}
        for key in ("embed_model", "title_inject"):
            if key in meta and meta[key] != CFG[key]:
                raise SystemExit(f"index was built with {key}={meta[key]} but config says {CFG[key]}: run embed.py first")
        self.chunks = [json.loads(l) for l in open(CHUNKS)]
        self.index = faiss.read_index(INDEX)
        self.model = SentenceTransformer(CFG["embed_model"])
        self.bm25 = BM25Okapi([tokenize(index_text(c)) for c in self.chunks])

    def dense(self, q: str, k: int = 10) -> list[tuple[int, float]]:
        qv = self.model.encode([query_text(q)], normalize_embeddings=True).astype("float32")
        scores, idx = self.index.search(qv, k)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0])]

    def bm25_search(self, q: str, k: int = 10) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(q))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]

    def hybrid(self, q: str, k: int = 10) -> list[tuple[int, float]]:
        """Weighted reciprocal rank fusion of the dense and BM25 rankings."""
        pool, rrf_k = CFG["pool"], CFG["rrf_k"]
        fused: dict[int, float] = {}
        for ranking, w in ((self.dense(q, pool), 1.0), (self.bm25_search(q, pool), CFG["bm25_weight"])):
            for rank, (i, _) in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + w / (rrf_k + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]

    def candidates(self, q: str) -> list[dict]:
        """Pool handed to the reranker: fused top-pool, or the union of both lists when pool_union=1."""
        pool = CFG["pool"]
        if CFG["pool_union"]:
            seen: dict[int, float] = {}
            for i, s in self.dense(q, pool) + self.bm25_search(q, pool):
                seen.setdefault(i, s)
            ranked = list(seen.items())
        else:
            ranked = self.hybrid(q, pool)
        return [{**self.chunks[i], "score": s} for i, s in ranked]

    def search(self, q: str, mode: str = "hybrid", k: int = 10) -> list[dict]:
        fn = {"dense": self.dense, "bm25": self.bm25_search, "hybrid": self.hybrid}[mode]
        return [{**self.chunks[i], "score": s, "rank": r}
                for r, (i, s) in enumerate(fn(q, k), start=1)]