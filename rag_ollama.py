# rag_ollama.py — Ollama backed answerer with retrieval
import os
import textwrap
import requests
from typing import List, Dict, Any

from nfpa_qa import search  # reuse your built index + retrieval

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

SYSTEM_PROMPT = """You are a strict code-compliance assistant.
Answer ONLY using the provided context from NFPA 13 (2022) and PCI NFPA 13R.
- If the answer is not supported by the context, say you don’t find support in the provided pages.
- Always include page-number citations in-line like (NFPA 13-2022 p.X) or (PCI NFPA 13R p.Y).
- Be concise and precise. Do not speculate. Do not invent page numbers.
"""

def hits_to_context(hits: List[Dict[str, Any]], max_chars: int = 3500) -> str:
    buf, total = [], 0
    for h in hits:
        snippet = (h.get("text") or "").strip()
        if not snippet:
            continue
        header = f"[{h['doc']} p.{h['page']}]"
        block = f"{header}\n{snippet}\n"
        if total + len(block) > max_chars:
            break
        buf.append(block)
        total += len(block)
    return "\n---\n".join(buf)

def ask_ollama(q: str, k: int = 8) -> Dict[str, Any]:
    hits = search(q, k=k)
    if not hits:
        return {"answer": "I couldn’t find anything relevant in the provided PDFs.", "hits": []}

    context = hits_to_context(hits)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {q}\n\n"
                    "Answer using only the context above, with citations."
                ),
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
            "num_predict": 128,  # keep it snappy
        },
    }

    try:
        # client-side timeout so we never block forever at this layer
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=45)
        resp.raise_for_status()
        data = resp.json()
        answer_text = (data.get("message", {}) or {}).get("content", "") or ""
        answer_text = answer_text.strip() or "The model returned an empty response."
    except Exception as e:
        answer_text = (
            "(Ollama error) Falling back to extractive answer.\n\n"
            f"Error: {e}\n\n"
            f"Context:\n{textwrap.shorten(context, width=1200)}"
        )

    return {"answer": answer_text, "hits": hits}
