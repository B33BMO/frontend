# server.py — One FastAPI app for the Site API + Twilio SMS using the same RAG backend
# Run: uvicorn server:app --host 0.0.0.0 --port 8000
from __future__ import annotations

import os
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Optional, Any, List

from fastapi import FastAPI, BackgroundTasks, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from rag_ollama import ask_ollama  # your existing retrieval + Ollama call (blocking requests)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MAX_WORKERS = int(os.environ.get("RAG_WORKERS", "4"))
DEFAULT_K = int(os.environ.get("RAG_K", "6"))
JOB_TTL_SECONDS = int(os.environ.get("RAG_JOB_TTL", "3600"))  # keep results for 1h

TWILIO_SID   = os.environ.get("TWILIO_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN")
TWILIO_FROM  = os.environ.get("TWILIO_FROM")
SEND_FOLLOWUP = bool(TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM)

# -----------------------------------------------------------------------------
# App + CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="NFPA Site + SMS API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down to your site domains if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log = logging.getLogger("uvicorn.error")

# -----------------------------------------------------------------------------
# Job system (simple in-memory store)
# -----------------------------------------------------------------------------
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

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
    sync: bool = Field(False, description="If true, wait and return the answer in this call")

class JobStatus(BaseModel):
    id: str
    status: str  # queued, running, done, error
    created_at: float
    finished_at: Optional[float] = None
    result: Optional[AskResponse] = None
    error: Optional[str] = None

_jobs: Dict[str, JobStatus] = {}
_futures: Dict[str, Future] = {}

def _cleanup_jobs() -> None:
    now = time.time()
    stale = [jid for jid, j in _jobs.items() if now - j.created_at > JOB_TTL_SECONDS]
    for jid in stale:
        _jobs.pop(jid, None)
        _futures.pop(jid, None)

def _run_rag(q: str, k: int) -> AskResponse:
    data = ask_ollama(q, k=k)  # returns {"answer": str, "hits": [dict,...]}
    hits_p = [Hit(**h) for h in data.get("hits", [])]
    return AskResponse(answer=data.get("answer", ""), hits=hits_p)

def _submit_job(q: str, k: int) -> str:
    _cleanup_jobs()
    jid = uuid.uuid4().hex
    _jobs[jid] = JobStatus(id=jid, status="queued", created_at=time.time())
    def worker():
        try:
            _jobs[jid].status = "running"
            result = _run_rag(q, k)
            _jobs[jid].result = result
            _jobs[jid].status = "done"
            _jobs[jid].finished_at = time.time()
        except Exception as e:
            log.exception("Job failed")
            _jobs[jid].status = "error"
            _jobs[jid].error = str(e)
            _jobs[jid].finished_at = time.time()
    _futures[jid] = executor.submit(worker)
    return jid

# -----------------------------------------------------------------------------
# (Optional) Twilio client for SMS follow-up
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
    body = (body or "").strip()
    if not body:
        return ["(empty)"]
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
# Health
# -----------------------------------------------------------------------------
@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "workers": MAX_WORKERS,
        "followup_enabled": SEND_FOLLOWUP,
        "inflight": len([j for j in _jobs.values() if j.status in ("queued", "running")]),
    }

# -----------------------------------------------------------------------------
# Website API
# -----------------------------------------------------------------------------
@app.post("/api/ask", response_model=JobStatus)
def api_ask(body: AskRequest):
    """
    Default: async job (returns job_id immediately).
    Pass sync=true to wait and return the full answer.
    """
    q = body.q.strip()
    k = body.k

    if not q:
        return JSONResponse({"detail": "q is required"}, status_code=400)

    if body.sync:
        try:
            result = _run_rag(q, k)
            return JobStatus(
                id="sync",
                status="done",
                created_at=time.time(),
                finished_at=time.time(),
                result=result,
            )
        except Exception as e:
            log.exception("sync ask failed")
            return JobStatus(
                id="sync",
                status="error",
                created_at=time.time(),
                finished_at=time.time(),
                error=str(e),
            )

    jid = _submit_job(q, k)
    return _jobs[jid]

@app.get("/api/result", response_model=JobStatus)
def api_result(id: str = Query(..., description="job_id from /api/ask")):
    j = _jobs.get(id)
    if not j:
        return JSONResponse({"detail": "job not found"}, status_code=404)
    return j

# -----------------------------------------------------------------------------
# Twilio SMS webhook (same engine, background; instant ACK)
# -----------------------------------------------------------------------------
@app.get("/sms", response_class=PlainTextResponse)
def sms_probe():
    return PlainTextResponse("<Response><Message>GET OK</Message></Response>",
                             media_type="application/xml")

@app.post("/sms", response_class=PlainTextResponse)
def sms_webhook(
    background: BackgroundTasks,
    From: str = Form(""),
    Body: str = Form(""),
):
    q = (Body or "").strip()
    dest = (From or "").strip()

    if not q:
        return PlainTextResponse(
            "<Response><Message>Send me a question.</Message></Response>",
            media_type="application/xml",
        )

    # queue the same RAG job
    jid = _submit_job(q, DEFAULT_K)

    # also schedule SMS follow-up using that job’s result
    def wait_and_text():
        # busy-wait in this thread until done (simple & fine here)
        fut = _futures.get(jid)
        if fut:
            try:
                fut.result()  # blocks until job completes
            except Exception:
                pass
        j = _jobs.get(jid)
        if j and j.status == "done" and dest:
            _send_sms(dest, j.result.answer)

    background.add_task(wait_and_text)

    # instant ACK for Twilio
    return PlainTextResponse(
        "<Response><Message>Got it — working on it…</Message></Response>",
        media_type="application/xml",
    )
