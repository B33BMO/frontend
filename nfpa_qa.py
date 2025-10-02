#!/usr/bin/env python3
"""
nfpa_qa.py — tiny local RAG over:
  - NFPA 13-2022 (edufire mirror)
  - PCI NFPA 13R PDF

Features
- Per-page parsing, chunking with overlap, page-aware citations
- Embeddings:
    * If OPENAI_API_KEY is set -> use OpenAI text-embedding-3-small (no torch).
    * Else -> use sentence-transformers all-MiniLM-L6-v2 (requires torch).
- Retrieval: cosine similarity (NumPy) + BM25 (rank-bm25) hybrid
- Exports:
    search(q, k=8) -> [{doc, page, text, score}]
    format_answer(q, hits) -> stitched extractive answer + inline citations
- CLI:
    python nfpa_qa.py --build
    python nfpa_qa.py --ask "When are sprinklers required on exterior balconies?"
"""

from __future__ import annotations
import os, sys, json, re, pathlib, textwrap, argparse, shutil
from typing import List, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm

# ----------- Paths / Config -----------
BASE = pathlib.Path(__file__).parent.resolve()
DATA = BASE / "data"
STORE = BASE / "store"
DATA.mkdir(exist_ok=True)
STORE.mkdir(exist_ok=True)

# Source PDFs
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

# Chunking
MAX_CHARS = 1200
OVERLAP   = 200

# Embedding mode
USE_OPENAI_EMB = bool(os.environ.get("OPENAI_API_KEY"))
OPENAI_MODEL   = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Store files
EMBS_NPY   = STORE / "embs.npy"
CORPUS_JSON= STORE / "corpus.json"
BM25_TXT   = STORE / "bm25_texts.json"


# ----------- Utilities -----------
def dl(url: str, out: pathlib.Path, timeout: int = 60) -> None:
    import requests
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def read_pdf_pages(pdf_path: pathlib.Path) -> List[str]:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        # tidy line breaks just a little
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

def save_json(path: pathlib.Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


# ----------- Embeddings -----------
def embed_openai(texts: List[str]) -> np.ndarray:
    """OpenAI embeddings (if OPENAI_API_KEY set) – no torch dependency."""
    import json, urllib.request
    key = os.environ["OPENAI_API_KEY"]
    url = "https://api.openai.com/v1/embeddings"

    # OpenAI accepts up to a decent batch; we’ll do simple 1 batch for simplicity
    body = json.dumps({"model": OPENAI_MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    embs = np.array([d["embedding"] for d in data["data"]], dtype="float32")
    # L2-normalize for cosine
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return embs

def embed_local(texts: List[str]) -> np.ndarray:
    """Local sentence-transformers embeddings (requires torch)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode(
        texts, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True
    )
    return embs.astype("float32")

def embed_texts(texts: List[str]) -> np.ndarray:
    if USE_OPENAI_EMB:
        return embed_openai(texts)
    return embed_local(texts)


# ----------- Build / Load index -----------
def build_index() -> None:
    print("Building corpus…")
    corpus = build_corpus()
    texts  = [c["text"] for c in corpus]
    print(f"Total chunks: {len(texts)}")

    print("Embedding…")
    embs = embed_texts(texts)  # (N, D)
    assert embs.shape[0] == len(texts), "Embeddings count mismatch"

    print("Saving store…")
    np.save(EMBS_NPY, embs)
    save_json(CORPUS_JSON, corpus)
    save_json(BM25_TXT, texts)
    print("Index built ✅")

def load_index() -> Tuple[List[Dict[str, Any]], List[str], np.ndarray]:
    if not (EMBS_NPY.exists() and CORPUS_JSON.exists() and BM25_TXT.exists()):
        raise RuntimeError("Index not built. Run: python nfpa_qa.py --build")
    corpus = load_json(CORPUS_JSON)
    texts  = load_json(BM25_TXT)
    embs   = np.load(EMBS_NPY)
    return corpus, texts, embs


# ----------- Search -----------
def _semantic_search(q: str, texts: List[str], embs: np.ndarray, topk: int) -> List[Tuple[int, float]]:
    qvec = embed_texts([q])[0]          # (D,)
    sims = embs @ qvec                  # cosine since embs are normalized
    idxs = np.argsort(-sims)[:topk]
    return [(int(i), float(sims[i])) for i in idxs]

def _bm25_scores(q: str, texts: List[str], topk: int) -> List[Tuple[int, float]]:
    from rank_bm25 import BM25Okapi
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    top_idx = bm25.get_top_n(q.lower().split(), list(range(len(texts))), n=topk)
    # pseudo scores: higher for earlier rank
    return [(int(i), float(topk - r)) for r, i in enumerate(top_idx)]

def search(query: str, k: int = 8) -> List[Dict[str, Any]]:
    """Hybrid retrieval: cosine + BM25 with simple additive fusion."""
    corpus, texts, embs = load_index()

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


# ----------- Answer stitcher (extractive fallback) -----------
def format_answer(q: str, hits: List[Dict[str, Any]]) -> str:
    """
    Simple extractive answer used as a fallback (and for quick CLI).
    Ollama path (rag_ollama.py) will do the generative composition instead.
    """
    if not hits:
        return "I couldn’t find anything relevant in the two PDFs."

    # stitch top passages, prefer ones that include query terms
    key_terms = re.sub(r"[^a-z0-9 ]+", " ", q.lower()).split()
    passages = [h["text"] for h in hits[:6]]
    stitched = " ".join(passages)
    # Trim stitched to avoid massive blobs
    stitched = (stitched[:2200] + "…") if len(stitched) > 2200 else stitched

    keep: List[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", stitched):
        score = sum(sent.lower().count(w) for w in key_terms)
        if score > 0 or len(keep) < 3:
            keep.append(sent)
        if len(keep) >= 6:
            break

    draft = " ".join(keep).strip()
    cites = []
    for h in hits[:4]:
        cites.append(f"({h['doc']} p.{h['page']})")
    cites_str = ", ".join(sorted(set(cites), key=cites.index))
    return f"{draft}\n\nSources: {cites_str}"


# ----------- CLI -----------
def _run_build() -> None:
    try:
        build_index()
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)

def _run_query(q: str, k: int) -> None:
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

def main() -> None:
    ap = argparse.ArgumentParser(description="Tiny NFPA 13/13R QA (index + hybrid retrieval)")
    ap.add_argument("--build", action="store_true", help="(re)build the local index")
    ap.add_argument("--ask", type=str, help="ask a question (extractive fallback answer)")
    ap.add_argument("-k", type=int, default=8, help="top-k to retrieve")
    args = ap.parse_args()

    if args.build:
        _run_build()
    if args.ask:
        _run_query(args.ask, k=args.k)
    if not args.build and not args.ask:
        print("Usage:\n  python nfpa_qa.py --build\n  python nfpa_qa.py --ask \"Exterior balcony sprinklers?\"")

if __name__ == "__main__":
    main()
