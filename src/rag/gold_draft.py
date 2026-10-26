# src/rag/gold_draft.py - Step 7 accelerator: LLM-drafted questions, human-verified
#   uv run src/rag/gold_draft.py 2406.13249v2 4   -> 4 random chunks of that paper -> eval/gold_draft.jsonl
import json, random, re, sys
from pathlib import Path
from openai import OpenAI
from generate import BASE_URL, API_KEY, MODEL

CHUNKS = "data/processed/chunks.jsonl"
DRAFT = Path("eval/gold_draft.jsonl")

PROMPT = PROMPT = """You write evaluation questions for a document-QA system.
Ask about findings, methods, datasets, or results. Never ask about table or figure numbers,
and never ask about bibliography entries. Make each question self-contained by naming the paper or method.
From the passage below, write exactly 2 questions:
1. type "factual": asks about one concrete fact in the passage.
2. type "paraphrase": asks about a fact using different words than the passage (no key term copied).
If the passage contains a number, make one of them type "numeric" instead.
For each question give:
- "evidence": 8 to 20 consecutive words COPIED EXACTLY from the passage (same spelling and punctuation) that contain the answer
- "answer": one sentence
Return ONLY a JSON list, no prose:
[{"type": "...", "question": "...", "evidence": "...", "answer": "..."}, ...]

Passage:
"""

def norm(s): return re.sub(r"\s+", " ", s.lower()).strip()

def draft(client, chunk):
    resp = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=1500,
        messages=[{"role": "user", "content": PROMPT + chunk["text"]}])
    text = resp.choices[0].message.content
    text = text[text.find("["): text.rfind("]") + 1]       # tolerate code fences / preamble
    return json.loads(text)

def main():
    doc_id = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    chunks = [json.loads(l) for l in open(CHUNKS)]
    mine = [c for c in chunks if c["doc_id"] == doc_id and not c["chunk_id"].endswith(":0")]
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    kept = dropped = 0
    with DRAFT.open("a", encoding="utf-8") as f:
        for c in random.sample(mine, min(n, len(mine))):
            try:
                items = draft(client, c)
            except Exception as e:
                print(f'{c["chunk_id"]}: LLM output not parseable ({e})'); continue
            for it in items:
                if norm(it.get("evidence", "")) in norm(c["text"]):
                    rec = {"qid": f'draft-{c["chunk_id"]}', "type": it["type"], "question": it["question"],
                           "doc_id": doc_id, "evidence": it["evidence"], "answer": it["answer"]}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); kept += 1
                    print(f'KEEP  {c["chunk_id"]}  [{it["type"]}] {it["question"]}')
                else:
                    dropped += 1
                    print(f'DROP  {c["chunk_id"]}  evidence not verbatim')
    print(f"\n{kept} drafts written to {DRAFT}, {dropped} dropped")

if __name__ == "__main__":
    main()