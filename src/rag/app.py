# src/rag/app.py - Step 9: FastAPI service around the pipeline
#   uv run uvicorn app:app --app-dir src/rag --port 7860
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()                                   # must run before generate.py reads the env
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from config import CFG
from retrieve import Retriever
from rerank import Reranker
from generate import Generator, MODEL

state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["r"], state["rr"], state["g"] = Retriever(), Reranker(CFG["rerank_model"]), Generator()
    yield
    state.clear()

app = FastAPI(title="RAG over 50 RAG papers", lifespan=lifespan)

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    k: int = Field(5, ge=1, le=10)

class SearchRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    mode: str = Field("hybrid", pattern="^(dense|bm25|hybrid|rerank)$")
    k: int = Field(10, ge=1, le=30)

def ctx(h: dict) -> dict:
    return {"chunk_id": h["chunk_id"], "title": h["title"], "page": h["page_start"],
            "score": round(float(h.get("rerank_score", h.get("score", 0.0))), 4), "text": h["text"][:400]}

@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["r"].chunks), "embed_model": CFG["embed_model"],
            "rerank_model": CFG["rerank_model"], "llm": MODEL}

@app.post("/search")
def search(req: SearchRequest):
    r, rr = state["r"], state["rr"]
    hits = (rr.rerank(req.question, r.candidates(req.question), req.k) if req.mode == "rerank"
            else r.search(req.question, req.mode, req.k))
    return {"question": req.question, "mode": req.mode, "hits": [ctx(h) for h in hits]}

@app.post("/ask")
def ask(req: AskRequest):
    t0 = time.time()
    r, rr, g = state["r"], state["rr"], state["g"]
    top = rr.rerank(req.question, r.candidates(req.question), req.k)
    try:
        out = g.answer(req.question, top)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")
    return {"question": req.question, "answer": out["answer"], "cited": out["cited"],
            "contexts": [ctx(h) for h in top], "latency_ms": int((time.time() - t0) * 1000), "llm": out["model"]}

INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>RAG over 50 RAG papers</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font-family:system-ui,sans-serif;max-width:800px;margin:2rem auto;padding:0 1rem;line-height:1.5}
input{width:100%;padding:.6rem;font-size:1rem}button{margin-top:.5rem;padding:.5rem 1rem;font-size:1rem}
.ctx{border-left:3px solid #999;padding:.3rem .8rem;margin:.6rem 0;font-size:.9rem;color:#444}
#answer{margin-top:1rem;white-space:pre-wrap}</style></head><body>
<h2>Ask the papers</h2><p>50 arXiv papers on retrieval-augmented generation. Hybrid retrieval, cross-encoder reranking, cited answers.</p>
<input id="q" placeholder="e.g. Which metrics does RAGAS propose?"><button onclick="ask()">Ask</button>
<div id="answer"></div><div id="ctx"></div>
<script>
async function ask(){const q=document.getElementById('q').value;document.getElementById('answer').textContent='Thinking...';document.getElementById('ctx').innerHTML='';
const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
const d=await r.json();if(!r.ok){document.getElementById('answer').textContent='Error: '+(d.detail||r.status);return;}
document.getElementById('answer').textContent=d.answer+'\\n\\n('+d.latency_ms+' ms)';
document.getElementById('ctx').innerHTML=d.contexts.map(c=>`<div class="ctx"><b>[${c.chunk_id}]</b> ${c.title} (p.${c.page})<br>${c.text}...</div>`).join('');}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')ask();});
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML