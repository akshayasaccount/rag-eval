# src/rag/gold_review.py - interactive review: drafts -> gold.jsonl
#   uv run src/rag/gold_review.py          review drafts one at a time
#   uv run src/rag/gold_review.py prune    drop gold lines whose evidence span matches no chunk
import json, re, sys
from pathlib import Path

CHUNKS = Path("data/processed/chunks.jsonl")
GOLD = Path("eval/gold.jsonl")
DRAFT = Path("eval/gold_draft.jsonl")

def norm(s): return re.sub(r"\s+", " ", s.lower()).strip()
def load(p): return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []
def save(p, rows):
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def next_qid(gold):
    nums = [int(g["qid"][1:]) for g in gold if g["qid"][1:].isdigit()]
    return f"q{(max(nums) if nums else 0) + 1:03d}"

def review():
    gold, drafts, remaining = load(GOLD), load(DRAFT), []
    for i, d in enumerate(drafts, 1):
        print(f"\n--- draft {i}/{len(drafts)}  [{d['type']}]  {d['doc_id']}")
        print("Q:", d["question"]); print("evidence:", d["evidence"]); print("A:", d["answer"])
        choice = ""
        while choice not in ("k", "e", "d", "s", "q"):
            choice = input("[k]eep  [e]dit then keep  [d]elete  [s]kip  [q]uit > ").strip().lower()
        if choice == "q":
            remaining += drafts[i - 1:]; break
        if choice == "d":
            continue
        if choice == "e":
            q = input("new question (Enter = keep): ").strip()
            a = input("new answer   (Enter = keep): ").strip()
            if q: d["question"] = q
            if a: d["answer"] = a
        if choice in ("k", "e"):
            d["qid"] = next_qid(gold); gold.append(d); print("   ->", d["qid"])
        else:
            remaining.append(d)
    save(GOLD, gold); save(DRAFT, remaining)
    print(f"\ngold.jsonl: {len(gold)} questions   drafts left: {len(remaining)}")

def prune():
    chunks, gold, kept = load(CHUNKS), load(GOLD), []
    for g in gold:
        if g.get("type") == "not-in-corpus" or any(
                c["doc_id"] == g["doc_id"] and norm(g["evidence"]) in norm(c["text"]) for c in chunks):
            kept.append(g)
        else:
            print("removing", g["qid"], "(span not found)")
    save(GOLD, kept); print(f"{len(kept)} questions kept")

if __name__ == "__main__":
    prune() if sys.argv[1:] == ["prune"] else review()