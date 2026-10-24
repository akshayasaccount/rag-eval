# src/rag/ingest.py — Step 1: arXiv PDFs -> data/processed/docs.jsonl
import json, re, time, urllib.request
from pathlib import Path
import arxiv, pymupdf

QUERY = 'all:"retrieval augmented generation"'   # your theme
N_PAPERS = 50
RAW = Path("data/raw")
OUT = Path("data/processed/docs.jsonl")

def clean(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)   # re-join "retrie-\nval"
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def extract_pages(pdf: Path) -> list[dict]:
    with pymupdf.open(pdf) as doc:
        return [{"page": i + 1, "text": clean(page.get_text())}
                for i, page in enumerate(doc)]

def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client()                       # 3 s delay built in
    search = arxiv.Search(query=QUERY, max_results=N_PAPERS,
                          sort_by=arxiv.SortCriterion.Relevance)
    with OUT.open("w") as f:
        for r in client.results(search):
            pdf = RAW / f"{r.get_short_id()}.pdf"
            if not pdf.exists():
                urllib.request.urlretrieve(r.pdf_url, pdf)
                time.sleep(3)                     # be polite to arXiv
            rec = {
                "doc_id": r.get_short_id(),
                "title": r.title,
                "authors": [a.name for a in r.authors],
                "published": r.published.date().isoformat(),
                "url": r.entry_id,
                "pages": extract_pages(pdf),
            }
            f.write(json.dumps(rec) + "\n")
            print("ok", rec["doc_id"], len(rec["pages"]), "pages")

if __name__ == "__main__":
    main()