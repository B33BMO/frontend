# api.py — FastAPI wrapper (Ollama answerer + retrieval)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from rag_ollama import ask_ollama  # <— actually used now

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

class AskRequest(BaseModel):
    q: str
    k: int = 3

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/ask", response_model=AskResponse)
def ask_get(q: str, k: int = 3):
    return ask_ollama(q, k=k)

@app.post("/api/ask", response_model=AskResponse)
def ask_post(body: AskRequest):
    return ask_ollama(body.q, k=body.k)
