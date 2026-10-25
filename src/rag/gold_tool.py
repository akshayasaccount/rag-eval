# src/rag/gold_tool.py - helpers for writing the gold set
#   uv run src/rag/gold_tool.py list                  all papers: doc_id, #chunks, title
#   uv run src/rag/gold_tool.py show 2309.15217v2     one paper's chunks: id, pages, first 200 chars
#   uv run src/rag/gold_tool.py find answer relevance chunks containing a phrase
#   uv run src/rag/gold_tool.py check                 validate eval/gold.jsonl: every evidence span matches a chunk
import json, re, sys
from collections import Counter

CHUNKS = "data/processed/chunks.jsonl"
GOLD = "eval/gold.jsonl"

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

chunks = [json.loads(l) for l in open(CHUNKS)]

def cmd_list():
    per = Counter(c["doc_id"] for c in chunks)
    titles = {c["doc_id"]: c["title"] for c in chunks}
    for d, n in sorted(per.items()):
        print(f"{d:16} {n:4}  {titles[d][:70]}")

def cmd_show(doc_id):
    for c in chunks:
        if c["doc_id"] == doc_id:
            print(f'\n[{c["chunk_id"]}] p{c["page_start"]}-{c["page_end"]}\n{c["text"][:200]}')

def cmd_find(phrase):
    p = norm(phrase)
    for c in chunks:
        t = norm(c["text"])
        if p in t:
            i = t.index(p)
            print(f'[{c["chunk_id"]}] p{c["page_start"]}  ...{t[max(0, i - 80):i + 120]}...')

def cmd_check():
    ok = bad = skipped = 0
    for n, line in enumerate(open(GOLD, encoding="utf-8"), start=1):
        if not line.strip():
            continue
        try:
            g = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"line {n}: BROKEN JSON ({e.msg})"); bad += 1; continue
        if g.get("type") == "not-in-corpus":
            skipped += 1; continue
        hits = [c["chunk_id"] for c in chunks
                if c["doc_id"] == g["doc_id"] and norm(g["evidence"]) in norm(c["text"])]
        if hits:
            ok += 1; print(f'{g["qid"]}  ok   {hits}')
        else:
            bad += 1; print(f'{g["qid"]}  MISSING  no chunk of {g["doc_id"]} contains that span')
    print(f"\n{ok} ok, {bad} problems, {skipped} not-in-corpus")

if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["list"]
    {"list": cmd_list, "show": lambda: cmd_show(args[0]),
     "find": lambda: cmd_find(" ".join(args)), "check": cmd_check}[cmd]()