# sms_ai.py — Twilio-safe SMS webhook (instant ACK + background answer)
from __future__ import annotations

import os
import time
import logging
from typing import Optional, Dict, List

from fastapi import FastAPI, Form, BackgroundTasks
from fastapi.responses import PlainTextResponse, JSONResponse

from rag_ollama import ask_ollama  # blocking requests under the hood; that's fine in a thread

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
TWILIO_SID: Optional[str]   = os.environ.get("TWILIO_SID")
TWILIO_TOKEN: Optional[str] = os.environ.get("TWILIO_TOKEN")
TWILIO_FROM: Optional[str]  = os.environ.get("TWILIO_FROM")  # e.g. "+1833XXXXXXX"
SEND_FOLLOWUP: bool         = bool(TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM)

# Retrieval size (tweak as you like)
DEFAULT_K = int(os.environ.get("SMS_K", "6"))

# Simple per-number cooldown (seconds)
COOLDOWN_SECONDS = int(os.environ.get("SMS_COOLDOWN_SECONDS", "3"))

# Optional max segments to send (each ~153 chars for long SMS concatenation)
MAX_SEGMENTS = int(os.environ.get("SMS_MAX_SEGMENTS", "6"))

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="NFPA SMS Bot", version="1.0.0")

# In-memory "last asked at" tracker (best-effort; resets on restart)
_last_asked_at: Dict[str, float] = {}

# Twilio client (optional)
_twilio_client = None
if SEND_FOLLOWUP:
    try:
        from twilio.rest import Client  # type: ignore
        _twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
        logger.info("Twilio follow-up SMS is ENABLED.")
    except Exception as e:
        logger.error(f"Failed to init Twilio client: {e}")
        _twilio_client = None
        SEND_FOLLOWUP = False
else:
    logger.info("Twilio follow-up SMS is DISABLED (missing env).")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _cooldown_hit(phone: str) -> bool:
    now = time.time()
    last = _last_asked_at.get(phone, 0.0)
    if now - last < COOLDOWN_SECONDS:
        return True
    _last_asked_at[phone] = now
    return False


def _chunk_for_sms(text: str, max_segments: int = MAX_SEGMENTS) -> List[str]:
    """
    Rough, GSM-safe-ish chunking. Carriers typically concat segments of ~153 chars.
    We keep it simple: 150 chars per segment, cap segments.
    """
    if not text:
        return ["(empty)"]
    text = text.strip()
    seg_size = 150
    chunks = [text[i:i+seg_size] for i in range(0, len(text), seg_size)]
    if len(chunks) > max_segments:
        chunks = chunks[:max_segments - 1] + ["[truncated]"]
    return chunks


def _send_followup_sms(to_number: str, body: str) -> None:
    if not SEND_FOLLOWUP or not _twilio_client:
        logger.info("[BG] Twilio not configured; skipping follow-up SMS.")
        return
    for part in _chunk_for_sms(body):
        _twilio_client.messages.create(
            to=to_number,
            from_=TWILIO_FROM,
            body=part,
        )


def _answer_and_optionally_sms(question: str, dest_number: Optional[str]) -> None:
    """
    Blocking work (runs in FastAPI's background thread).
    """
    try:
        result = ask_ollama(question, k=DEFAULT_K)  # returns {"answer": str, "hits": [...]}
        answer = (result.get("answer") or "").strip()
        if not answer:
            answer = "I couldn't produce an answer from the provided PDFs."
        logger.info(f"[BG] Answer ready ({len(answer)} chars) for {dest_number}: {question!r}")

        if dest_number:
            _send_followup_sms(dest_number, answer)
    except Exception as e:
        logger.exception(f"[BG] Error generating/sending answer: {e}")
        if dest_number:
            try:
                _send_followup_sms(dest_number, "Sorry, I hit an error answering that. Try again shortly.")
            except Exception:
                pass


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "followup_enabled": SEND_FOLLOWUP, "k": DEFAULT_K}

# Simple GET to test from a browser
@app.get("/sms", response_class=PlainTextResponse)
def sms_probe():
    return PlainTextResponse(
        "<Response><Message>GET OK</Message></Response>",
        media_type="application/xml",
    )

@app.post("/sms", response_class=PlainTextResponse)
def sms_webhook(
    background: BackgroundTasks,
    From: str = Form(""),
    Body: str = Form(""),
):
    """Twilio webhook: ACK instantly, do heavy work in background."""
    q = (Body or "").strip()
    dest = (From or "").strip()

    if not q:
        return PlainTextResponse(
            "<Response><Message>Send me a question.</Message></Response>",
            media_type="application/xml",
        )

    # rate-limit: quick & dirty
    if dest and _cooldown_hit(dest):
        return PlainTextResponse(
            "<Response><Message>Easy tiger—give me a couple seconds between texts.</Message></Response>",
            media_type="application/xml",
        )

    # Schedule blocking work in a thread so the loop replies immediately.
    background.add_task(_answer_and_optionally_sms, q, dest)

    # Short ACK for Twilio (must return within ~15s)
    ack = "Got it — working on it…" if SEND_FOLLOWUP else "Got it — I’ll reply here once the server is configured to send SMS."
    return PlainTextResponse(
        f"<Response><Message>{ack}</Message></Response>",
        media_type="application/xml",
    )
