# src/rag/check.py — eyeball Step 1 output
import json, random
docs = [json.loads(l) for l in open("data/processed/docs.jsonl")]
print(len(docs), "papers,", sum(len(d["pages"]) for d in docs), "pages")
empty = [d["doc_id"] for d in docs if sum(len(p["text"]) for p in d["pages"]) < 2000]
print("suspiciously empty:", empty)
for d in random.sample(docs, 3):
    print(f'\n{d["doc_id"]}  {d["title"][:55]}  {len(d["pages"])}p')
    print(d["pages"][1]["text"][:600], "\n---")