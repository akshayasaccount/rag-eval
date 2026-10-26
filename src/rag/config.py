# src/rag/config.py - every knob in one place, overridable with RAG_* environment variables
import os

def _env(name, default, cast=str):
    v = os.getenv(name)
    return default if v is None else cast(v)

CFG = {
    "embed_model":  _env("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
    "rerank_model": _env("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    "title_inject": _env("RAG_TITLE_INJECT", 0, int),   # prepend paper title to chunk text for embed + BM25
    "query_prefix": _env("RAG_QUERY_PREFIX", 0, int),   # bge instruction prefix on queries (dense only)
    "pool":         _env("RAG_POOL", 30, int),          # candidates per retriever before fusion / rerank
    "pool_union":   _env("RAG_POOL_UNION", 0, int),     # rerank the union of both lists instead of fused top-pool
    "bm25_weight":  _env("RAG_BM25_WEIGHT", 1.0, float),# RRF weight of the BM25 list (dense list = 1.0)
    "rrf_k":        _env("RAG_RRF_K", 60, int),
}
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def index_text(c: dict) -> str:
    """Text used for embedding and BM25. Display text stays c['text']."""
    return f'{c["title"]}. {c["text"]}' if CFG["title_inject"] else c["text"]

def query_text(q: str) -> str:
    return QUERY_PREFIX + q if CFG["query_prefix"] else q