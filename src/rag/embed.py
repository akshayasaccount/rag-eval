# src/rag/embed.py - Step 3: chunks.jsonl -> embeddings.npy + index.faiss
import json
from pathlib import Path
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 384-dim, CPU-friendly; ablate later
CHUNKS = Path("data/processed/chunks.jsonl")
EMB = Path("data/processed/embeddings.npy")
INDEX = Path("data/processed/index.faiss")
IDS = Path("data/processed/chunk_ids.json")

def main() -> None:
    chunks = [json.loads(l) for l in CHUNKS.open()]
    texts = [c["text"] for c in chunks]
    model = SentenceTransformer(MODEL_NAME)
    vecs = model.encode(texts, batch_size=32, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])   # exact cosine search on unit vectors
    index.add(vecs)
    np.save(EMB, vecs)
    faiss.write_index(index, str(INDEX))
    IDS.write_text(json.dumps([c["chunk_id"] for c in chunks]))
    print(f"{len(chunks)} chunks -> {vecs.shape} embeddings, index size {index.ntotal}")

if __name__ == "__main__":
    main()