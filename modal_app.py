# modal_app.py - deploy the FastAPI service on Modal (scales to zero when idle)
#   uv run modal deploy modal_app.py
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements("requirements-serve.txt")
    .run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer, CrossEncoder; "
        "SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')\""
    )
    .add_local_dir("src/rag", remote_path="/root/app/src/rag")
    .add_local_file("data/processed/chunks.jsonl", remote_path="/root/app/data/processed/chunks.jsonl")
    .add_local_file("data/processed/index.faiss", remote_path="/root/app/data/processed/index.faiss")
    .add_local_file("data/processed/index_meta.json", remote_path="/root/app/data/processed/index_meta.json")
)

app = modal.App("rag-eval", image=image)

@app.function(secrets=[modal.Secret.from_name("rag-eval-llm")], memory=2048, cpu=1.0,
              min_containers=0, scaledown_window=300)
@modal.asgi_app()
def web():
    import os, sys
    os.chdir("/root/app")
    sys.path.insert(0, "/root/app/src/rag")
    from app import app as fastapi_app
    return fastapi_app