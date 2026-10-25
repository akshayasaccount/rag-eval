# src/rag/check_chunks.py - eyeball Step 2 output
import json, random, statistics as st
from collections import Counter
chunks = [json.loads(l) for l in open("data/processed/chunks.jsonl")]
per_doc = Counter(c["doc_id"] for c in chunks)
sizes = [c["n_words"] for c in chunks]
print(len(chunks), "chunks from", len(per_doc), "docs")
print("chunks/doc  min", min(per_doc.values()), "median", st.median(per_doc.values()), "max", max(per_doc.values()))
print("words/chunk min", min(sizes), "median", st.median(sizes), "max", max(sizes))
print("biggest doc:", per_doc.most_common(1))
for c in random.sample(chunks, 2):
    print(f'\n{c["chunk_id"]}  p{c["page_start"]}-{c["page_end"]}  {c["n_words"]}w')
    print(c["text"][:500], "\n---")