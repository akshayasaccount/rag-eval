# src/rag/chunk.py - Step 2: docs.jsonl -> chunks.jsonl
import json, re
from pathlib import Path

IN = Path("data/processed/docs.jsonl")
OUT = Path("data/processed/chunks.jsonl")
CHUNK_WORDS = 250        # ablate later: 120 / 250 / 400
OVERLAP_WORDS = 50
STRIP_REFERENCES = False  # ablate later

REF_HEADING = re.compile(r"^\s*(references|bibliography)\s*$", re.I | re.M)

def tidy(text: str) -> str:
    """Join single line breaks inside paragraphs; keep blank-line paragraph breaks."""
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()

def doc_words(doc: dict) -> list[tuple[str, int]]:
    """Flatten a doc into (word, page) pairs, cutting at References if it is in the 2nd half."""
    words, half = [], len(doc["pages"]) / 2
    for p in doc["pages"]:
        text = p["text"]
        if STRIP_REFERENCES and p["page"] > half:
            m = REF_HEADING.search(text)
            if m:
                words += [(w, p["page"]) for w in tidy(text[:m.start()]).split()]
                break
        words += [(w, p["page"]) for w in tidy(text).split()]
    return words

def chunk_doc(doc: dict):
    words = doc_words(doc)
    step = CHUNK_WORDS - OVERLAP_WORDS
    for i, start in enumerate(range(0, max(len(words), 1), step)):
        window = words[start:start + CHUNK_WORDS]
        if not window:
            break
        yield {
            "chunk_id": f'{doc["doc_id"]}:{i}',
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "page_start": window[0][1],
            "page_end": window[-1][1],
            "n_words": len(window),
            "text": " ".join(w for w, _ in window),
        }
        if start + CHUNK_WORDS >= len(words):
            break

def main() -> None:
    n_docs = n_chunks = 0
    with IN.open() as fin, OUT.open("w") as fout:
        for line in fin:
            doc = json.loads(line)
            n_docs += 1
            for c in chunk_doc(doc):
                fout.write(json.dumps(c) + "\n")
                n_chunks += 1
    print(f"{n_docs} docs -> {n_chunks} chunks  "
          f"(size={CHUNK_WORDS}, overlap={OVERLAP_WORDS}, strip_refs={STRIP_REFERENCES})")

if __name__ == "__main__":
    main()