# src/rag/check_embed.py - ask the index a few questions
import json, faiss
from sentence_transformers import SentenceTransformer

chunks = [json.loads(l) for l in open("data/processed/chunks.jsonl")]
index = faiss.read_index("data/processed/index.faiss")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

QUESTIONS = [
    "What metrics does RAGAS use to evaluate a RAG pipeline?",
    "How does R2AG incorporate retrieval information into generation?",
    "What is the difference between dense and sparse retrieval?",
]
for q in QUESTIONS:
    qv = model.encode([q], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(qv, 5)
    print("\nQ:", q)
    for s, i in zip(scores[0], idx[0]):
        c = chunks[i]
        print(f"  {s:.3f}  {c['chunk_id']:<18} p{c['page_start']}  {c['title'][:50]}")
        print("         ", c["text"][:120])