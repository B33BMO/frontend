# server.py — One FastAPI app for Site API + Twilio SMS
# Run: uvicorn server:app --host 0.0.0.0 --port 8500
from __future__ import annotations

import os
import time
import uuid
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Optional, List

from fastapi import FastAPI, BackgroundTasks, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from rag_ollama import ask_ollama

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MAX_WORKERS = int(os.environ.get("RAG_WORKERS", "4"))
DEFAULT_K = int(os.environ.get("RAG_K", "6"))
JOB_TTL_SECONDS = int(os.environ.get("RAG_JOB_TTL", "3600"))
ASK_TIMEOUT = int(os.environ.get("ASK_TIMEOUT", "50"))  # API hard timeout (sec)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

TWILIO_SID   = os.environ.get("TWILIO_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN")
TWILIO_FROM  = os.environ.get("TWILIO_FROM")
SEND_FOLLOWUP = bool(TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM)

# -----------------------------------------------------------------------------
# App + CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="NFPA Site + SMS API", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
log = logging.getLogger("uvicorn.error")

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class Hit(BaseModel):
    doc: str
    page: int
    text: str
    score: float

class AskResponse(BaseModel):
    answer: str
    hits: List[Hit] = Field(default_factory=list)

class AskRequest(BaseModel):
    q: str
    k: int = Field(DEFAULT_K, ge=1, le=20)
    sync: bool = Field(False)

class JobStatus(BaseModel):
    id: str
    status: str               # queued, running, done, error
    created_at: float
    finished_at: Optional[float] = None
    result: Optional[AskResponse] = None
    error: Optional[str] = None
    elapsed_sec: Optional[float] = None

_jobs: Dict[str, JobStatus] = {}
_futures: Dict[str, Future] = {}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _cleanup_jobs() -> None:
    now = time.time()
    stale = [jid for jid, j in _jobs.items() if now - j.created_at > JOB_TTL_SECONDS]
    for jid in stale:
        _jobs.pop(jid, None)
        _futures.pop(jid, None)

def _run_rag_sync(q: str, k: int) -> AskResponse:
    data = ask_ollama(q, k=k)
    hits = [Hit(**h) for h in data.get("hits", [])]
    return AskResponse(answer=data.get("answer", ""), hits=hits)

async def _run_rag_with_timeout(q: str, k: int) -> AskResponse:
    # Run blocking RAG inside thread + cap total time
    return await asyncio.wait_for(
        run_in_threadpool(lambda: _run_rag_sync(q, k)),
        timeout=ASK_TIMEOUT,
    )

def _submit_job(q: str, k: int) -> str:
    _cleanup_jobs()
    jid = uuid.uuid4().hex
    _jobs[jid] = JobStatus(id=jid, status="queued", created_at=time.time())
    def worker():
        t0 = time.time()
        try:
            _jobs[jid].status = "running"
            # Use the same timeout cap for background work
            result = asyncio.run(_run_rag_with_timeout(q, k))
            _jobs[jid].result = result
            _jobs[jid].status = "done"
            _jobs[jid].finished_at = time.time()
            _jobs[jid].elapsed_sec = _jobs[jid].finished_at - t0
        except asyncio.TimeoutError:
            _jobs[jid].status = "error"
            _jobs[jid].error = f"timeout after {ASK_TIMEOUT}s"
            _jobs[jid].finished_at = time.time()
        except Exception as e:
            log.exception("Job failed")
            _jobs[jid].status = "error"
            _jobs[jid].error = str(e)
            _jobs[jid].finished_at = time.time()
    _futures[jid] = executor.submit(worker)
    return jid

# -----------------------------------------------------------------------------
# (Optional) Twilio follow-up
# -----------------------------------------------------------------------------
_twilio_client = None
if SEND_FOLLOWUP:
    try:
        from twilio.rest import Client  # type: ignore
        _twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
        log.info("Twilio follow-up SMS ENABLED.")
    except Exception as e:
        log.error(f"Failed to init Twilio client: {e}")
        SEND_FOLLOWUP = False

def _chunk_sms(body: str, seg_len: int = 150, max_segments: int = 6) -> List[str]:
    body = (body or "").strip() or "(empty)"
    parts = [body[i:i+seg_len] for i in range(0, len(body), seg_len)]
    if len(parts) > max_segments:
        parts = parts[:max_segments - 1] + ["[truncated]"]
    return parts

def _send_sms(to: str, body: str) -> None:
    if not (SEND_FOLLOWUP and _twilio_client and TWILIO_FROM):
        return
    for p in _chunk_sms(body):
        _twilio_client.messages.create(to=to, from_=TWILIO_FROM, body=p)

# -----------------------------------------------------------------------------
# Startup warm-up (avoid first-call stalls)
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def warmup():
    try:
        import requests
        # tag list
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        # tiny chat to load the model
        requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
            timeout=15,
        )
        log.info("Ollama warm-up complete.")
    except Exception as e:
        log.warning(f"Ollama warm-up skipped: {e}")

# -----------------------------------------------------------------------------
# Health & debug
# -----------------------------------------------------------------------------
@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "workers": MAX_WORKERS,
        "followup_enabled": SEND_FOLLOWUP,
        "inflight": len([j for j in _jobs.values() if j.status in ("queued", "running")]),
        "ask_timeout": ASK_TIMEOUT,
        "model": OLLAMA_MODEL,
    }

@app.get("/api/debug/ollama")
def debug_ollama():
    import requests
    try:
        t = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
        return {"ok": True, "tags": t}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=503)

# -----------------------------------------------------------------------------
# Website API
# -----------------------------------------------------------------------------
@app.post("/api/ask", response_model=JobStatus)
async def api_ask(body: AskRequest):
    q = (body.q or "").strip()
    k = int(body.k or DEFAULT_K)
    if not q:
        return JSONResponse({"detail": "q is required"}, status_code=400)

    if body.sync:
        t0 = time.time()
        try:
            result = await _run_rag_with_timeout(q, k)
            t1 = time.time()
            return JobStatus(
                id="sync", status="done",
                created_at=t0, finished_at=t1,
                elapsed_sec=round(t1 - t0, 3),
                result=result,
            )
        except asyncio.TimeoutError:
            return JSONResponse({"detail": f"timeout after {ASK_TIMEOUT}s"}, status_code=504)
        except Exception as e:
            log.exception("sync ask failed")
            return JSONResponse({"detail": f"server error: {e}"}, status_code=500)

    jid = _submit_job(q, k)
    return _jobs[jid]

@app.get("/api/result", response_model=JobStatus)
def api_result(id: str = Query(..., description="job_id from /api/ask")):
    j = _jobs.get(id)
    if not j:
        return JSONResponse({"detail": "job not found"}, status_code=404)
    return j

# -----------------------------------------------------------------------------
# Twilio SMS webhook (instant ACK, work in background)
# -----------------------------------------------------------------------------
@app.get("/sms", response_class=PlainTextResponse)
def sms_probe():
    return PlainTextResponse("<Response><Message>GET OK</Message></Response>",
                             media_type="application/xml")

@app.post("/sms", response_class=PlainTextResponse)
def sms_webhook(background: BackgroundTasks, From: str = Form(""), Body: str = Form("")):
    q = (Body or "").strip()
    dest = (From or "").strip()
    if not q:
        return PlainTextResponse("<Response><Message>Send me a question.</Message></Response>",
                                 media_type="application/xml")

    jid = _submit_job(q, DEFAULT_K)

    def wait_and_text():
        fut = _futures.get(jid)
        if fut:
            try:
                fut.result(timeout=ASK_TIMEOUT + 5)
            except Exception:
                pass
        j = _jobs.get(jid)
        if j and j.status == "done" and dest:
            _send_sms(dest, j.result.answer)

    background.add_task(wait_and_text)
    return PlainTextResponse("<Response><Message>Got it — working on it…</Message></Response>",
                             media_type="application/xml")
