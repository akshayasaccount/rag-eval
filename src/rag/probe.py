# src/rag/probe.py — why did the search return nothing?
import logging, arxiv
logging.basicConfig(level=logging.INFO)
client = arxiv.Client()
for q in ['all:"retrieval augmented generation"', "all:electron"]:
    n = sum(1 for _ in client.results(arxiv.Search(query=q, max_results=5)))
    print(q, "->", n, "results")