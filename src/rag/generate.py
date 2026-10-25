# src/rag/generate.py - Step 6: answer a question from retrieved chunks, with citations
import os, re
from openai import OpenAI

# Default = Ollama's OpenAI-compatible endpoint. For a hosted API set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")

SYSTEM = """You answer questions about research papers using ONLY the provided context.
Rules:
- Cite every claim with the chunk id in square brackets, e.g. [2309.15217v2:4].
- If the context does not contain the answer, reply exactly: Not found in the provided papers.
- Be concise: 2-5 sentences."""

CITE = re.compile(r"\[([0-9]{4}\.[0-9]{4,5}v[0-9]+:[0-9]+)\]")

def build_prompt(question: str, chunks: list[dict]) -> str:
    ctx = "\n\n".join(f'[{c["chunk_id"]}] ({c["title"][:60]}, p.{c["page_start"]})\n{c["text"]}'
                      for c in chunks)
    return (f"<context>\n{ctx}\n</context>\n\n"
            f"Answer the question using only the context above. Output only the answer.\n"
            f"Question: {question}")

class Generator:
    def __init__(self):
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    def answer(self, question: str, chunks: list[dict]) -> dict:
        resp = self.client.chat.completions.create(
            model=MODEL, temperature=0, max_tokens=1000,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": build_prompt(question, chunks)}],
        )
        text = resp.choices[0].message.content.strip()
        return {"question": question, "answer": text,
                "cited": sorted(set(CITE.findall(text))),
                "context_ids": [c["chunk_id"] for c in chunks],
                "model": MODEL}