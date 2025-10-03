#!/usr/bin/env python3
"""
nfpa_qa.py — tiny local RAG index over:
  - NFPA 13-2022 (edufire mirror)
  - PCI NFPA 13R (PCI PDF)

Fast path tuned:
- RAM caches for corpus/embeddings/BM25 and lazy SentenceTransformer
- Hybrid retrieval (cosine + BM25)
- Optional OpenAI embeddings if OPENAI_API_KEY is set; else sentence-transformers

CLI:
  python nfpa_qa.py --build
  python nfpa_qa.py --ask "Are sprinklers required on exterior balconies?"
"""

from __future__ import annotations
import os, json, re, pathlib, argparse, textwrap, sys
from typing import List, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm

# ---------------- Paths / Config ----------------
BASE = pathlib.Path(__file__).parent.resolve()
DATA = BASE / "data"
STORE = BASE / "store"
DATA.mkdir(exist_ok=True); STORE.mkdir(exist_ok=True)

PDFS = [
    {
        "name": "NFPA 13-2022",
        "url": "https://edufire.ir/storage/Library/ETFA-ABI/NFPA/NFPA%2013-2022.pdf",
        "file": DATA / "NFPA_13_2022.pdf",
    },
    {
        "name": "PCI NFPA 13R",
        "url": "https://www.pci.org/pci_docs/Design_Resources/Building_Engineering_Resources/NFPA_13R.pdf",
        "file": DATA / "PCI_NFPA_13R.pdf",
    },
]

MAX_CHARS = 1200
OVERLAP   = 200

USE_OPENAI_EMB = bool(os.environ.get("OPENAI_API_KEY"))
OPENAI_MODEL   = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

EMBS_NPY     = STORE / "embs.npy"
CORPUS_JSON  = STORE / "corpus.json"
BM25_TXT     = STORE / "bm25_texts.json"

# ---------------- Hot caches ----------------
_INDEX: Tuple[List[Dict[str, Any]], List[str], np.ndarray] | None = None
_BM25 = None
_SENT_MODEL = None

# ---------------- Utils ----------------
def dl(url: str, out: pathlib.Path, timeout: int = 60) -> None:
    import requests
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk: f.write(chunk)

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def read_pdf_pages(pdf_path: pathlib.Path) -> List[str]:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = re.sub(r"[ \t]+\n", "\n", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
        pages.append(txt)
    return pages

def chunkify(pages: List[str], max_chars=MAX_CHARS, overlap=OVERLAP) -> List[Tuple[int, str]]:
    chunks: List[Tuple[int, str]] = []
    for pnum, page_text in enumerate(pages, start=1):
        text = page_text or ""
        if len(text) <= max_chars:
            if text.strip():
                chunks.append((pnum, text))
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            chunk = text[start:end]
            if chunk.strip():
                chunks.append((pnum, chunk))
            if end == len(text): break
            start = max(0, end - overlap)
    return chunks

def save_json(path: pathlib.Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())

# ---------------- Build / Load ----------------
def build_corpus() -> List[Dict[str, Any]]:
    corpus: List[Dict[str, Any]] = []
    for meta in PDFS:
        if not meta["file"].exists():
            print(f"Downloading {meta['name']}…")
            dl(meta["url"], meta["file"])
        print(f"Parsing {meta['name']}…")
        pages = read_pdf_pages(meta["file"])
        for pnum, chunk in chunkify(pages):
            corpus.append({
                "doc": meta["name"],
                "page": pnum,
                "text": normalize_space(chunk),
                "source_path": str(meta["file"]),
            })
    return corpus

def build_index() -> None:
    corpus = build_corpus()
    texts  = [c["text"] for c in corpus]
    print(f"Total chunks: {len(texts)}")

    print("Embedding…")
    embs = embed_texts(texts)
    embs = embs.astype("float32")
    print("Saving store…")
    np.save(EMBS_NPY, embs)
    save_json(CORPUS_JSON, corpus)
    save_json(BM25_TXT, texts)
    print("Index built ✅")

def load_index_cached() -> Tuple[List[Dict[str, Any]], List[str], np.ndarray]:
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    if not (EMBS_NPY.exists() and CORPUS_JSON.exists() and BM25_TXT.exists()):
        raise RuntimeError("Index not built. Run: python nfpa_qa.py --build")
    corpus = load_json(CORPUS_JSON)
    texts  = load_json(BM25_TXT)
    embs   = np.load(EMBS_NPY)
    _INDEX = (corpus, texts, embs)
    return _INDEX

# ---------------- Embeddings ----------------
def embed_openai(texts: List[str]) -> np.ndarray:
    import json, urllib.request
    key = os.environ["OPENAI_API_KEY"]
    url = "https://api.openai.com/v1/embeddings"
    body = json.dumps({"model": OPENAI_MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    embs = np.array([d["embedding"] for d in data["data"]], dtype="float32")
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return embs

def embed_local(texts: List[str]) -> np.ndarray:
    global _SENT_MODEL
    if _SENT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SENT_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = _SENT_MODEL.encode(
        texts, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True
    )
    return embs.astype("float32")

def embed_texts(texts: List[str]) -> np.ndarray:
    if USE_OPENAI_EMB:
        return embed_openai(texts)
    return embed_local(texts)

# ---------------- Retrieval ----------------
def _semantic_search(q: str, texts: List[str], embs: np.ndarray, topk: int) -> List[Tuple[int, float]]:
    qvec = embed_texts([q])[0]
    sims = embs @ qvec  # cosine because normalized
    idxs = np.argsort(-sims)[:topk]
    return [(int(i), float(sims[i])) for i in idxs]

def _get_bm25(texts: List[str]):
    global _BM25
    if _BM25 is None:
        from rank_bm25 import BM25Okapi
        tokenized = [t.lower().split() for t in texts]
        _BM25 = BM25Okapi(tokenized)
    return _BM25

def _bm25_scores(q: str, texts: List[str], topk: int) -> List[Tuple[int, float]]:
    bm25 = _get_bm25(texts)
    top_idx = bm25.get_top_n(q.lower().split(), list(range(len(texts))), n=topk)
    return [(int(i), float(topk - r)) for r, i in enumerate(top_idx)]

def search(query: str, k: int = 8) -> List[Dict[str, Any]]:
    corpus, texts, embs = load_index_cached()
    sem_hits = _semantic_search(query, texts, embs, k)
    bm_hits  = _bm25_scores(query, texts, k)

    score_map: Dict[int, float] = {}
    for i, s in sem_hits + bm_hits:
        score_map[i] = score_map.get(i, 0.0) + s

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:k]
    results: List[Dict[str, Any]] = []
    for idx, score in ranked:
        c = corpus[idx]
        results.append({
            "doc": c["doc"],
            "page": c["page"],
            "text": c["text"],
            "score": round(float(score), 4),
        })
    return results

# ---------------- Extractive fallback ----------------
def format_answer(q: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "I couldn’t find anything relevant in the two PDFs."

    key_terms = re.sub(r"[^a-z0-9 ]+", " ", q.lower()).split()
    passages = [h["text"] for h in hits[:6]]
    stitched = " ".join(passages)
    stitched = (stitched[:2200] + "…") if len(stitched) > 2200 else stitched

    keep: List[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", stitched):
        score = sum(sent.lower().count(w) for w in key_terms)
        if score > 0 or len(keep) < 3:
            keep.append(sent)
        if len(keep) >= 6:
            break

    cites = [f"({h['doc']} p.{h['page']})" for h in hits[:4]]
    cites_str = ", ".join(sorted(set(cites), key=cites.index))
    return f"{' '.join(keep).strip()}\n\nSources: {cites_str}"


# ---------------- CLI ----------------
def _run_build():
    try:
        build_index()
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)

def _run_query(q: str, k: int):
    try:
        hits = search(q, k=k)
    except Exception as e:
        print(f"Search failed: {e}")
        sys.exit(1)
    ans = format_answer(q, hits)
    print("\nQ:", q)
    print("\nA:", textwrap.fill(ans, width=100))
    print("\nTop hits:")
    for r in hits[:5]:
        snippet = textwrap.shorten(r["text"], width=120)
        print(f" - {r['doc']} p.{r['page']}  score={r['score']}\n   {snippet}")

def main():
    ap = argparse.ArgumentParser(description="NFPA 13/13R RAG index")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ask", type=str)
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args()

    if args.build:
        _run_build()
    if args.ask:
        _run_query(args.ask, k=args.k)
    if not args.build and not args.ask:
        print("Usage:\n  python nfpa_qa.py --build\n  python nfpa_qa.py --ask \"Exterior balcony sprinklers?\"")

if __name__ == "__main__":
    main()
