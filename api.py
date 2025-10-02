# api.py — FastAPI wrapper (Ollama answerer + your retrieval)
# Run: uvicorn api:app --reload --port 8000

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from rag_ollama import ask_ollama  # <— NEW

app = FastAPI(title="NFPA Q&A API (Ollama)", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class Hit(BaseModel):
    doc: str
    page: int
    text: str
    score: float

class AskResponse(BaseModel):
    answer: str
    hits: List[Hit]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ask", response_model=AskResponse)
def ask(q: str = Query(..., min_length=3), k: int = 8):
    result = ask_ollama(q, k=k)
    # Shape normalization
    answer = result.get("answer", "")
    hits = result.get("hits", []) or []
    return {"answer": answer, "hits": hits}
