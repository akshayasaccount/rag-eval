FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface PIP_NO_CACHE_DIR=1
WORKDIR /home/user/app
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements-serve.txt .
RUN pip install --user -r requirements-serve.txt
COPY --chown=user src/rag ./src/rag
COPY --chown=user data/processed/chunks.jsonl data/processed/index.faiss data/processed/index_meta.json ./data/processed/
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
EXPOSE 7860
CMD ["uvicorn", "app:app", "--app-dir", "src/rag", "--host", "0.0.0.0", "--port", "7860"]
