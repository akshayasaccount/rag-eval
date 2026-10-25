# src/rag/check_strip.py - did STRIP_REFERENCES cut any paper too early?
import json
pages = {}
for l in open("data/processed/docs.jsonl"):
    d = json.loads(l); pages[d["doc_id"]] = len(d["pages"])
last = {}
for l in open("data/processed/chunks.jsonl"):
    c = json.loads(l); last[c["doc_id"]] = c["page_end"]
for doc_id, n in sorted(pages.items(), key=lambda x: last[x[0]] / x[1]):
    print(f"{doc_id}  pages={n:3d}  last_chunk_ends_p{last[doc_id]:3d}  kept={last[doc_id]/n:.0%}")