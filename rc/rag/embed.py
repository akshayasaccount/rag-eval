# src/rag/embed.py - Step 3/8: chunks.jsonl -> embeddings.npy + index.faiss (config-aware)
import json
from pathlib import Path
import numpy as np, faiss
from sentence_transformers import SentenceTransformer
from config import CFG, index_text

CHUNKS = Path("data/processed/chunks.jsonl")
EMB = Path("data/processed/embeddings.npy")
INDEX = Path("data/processed/index.faiss")
META = Path("data/processed/index_meta.json")

def main() -> None:
    chunks = [json.loads(l) for l in CHUNKS.open()]
    texts = [index_text(c) for c in chunks]
    model = SentenceTransformer(CFG["embed_model"])
    vecs = model.encode(texts, batch_size=32, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    np.save(EMB, vecs)
    faiss.write_index(index, str(INDEX))
    META.write_text(json.dumps({"embed_model": CFG["embed_model"],
                                "title_inject": CFG["title_inject"], "n": len(chunks)}))
    print(f"{len(chunks)} chunks -> {vecs.shape}  (model={CFG['embed_model']}, title_inject={CFG['title_inject']})")

if __name__ == "__main__":
    main()