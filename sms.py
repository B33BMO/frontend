# sms_api.py
from fastapi import FastAPI, Form
from fastapi.responses import PlainTextResponse
from rag_ollama import ask_ollama

app = FastAPI()

@app.post("/api/ask", response_class=PlainTextResponse)
async def sms_reply(From: str = Form(...), Body: str = Form(...)):
    """Webhook endpoint that Twilio calls when an SMS comes in."""
    query = Body.strip()
    result = ask_ollama(query, k=3)

    # Keep reply short (SMS has limits)
    answer = result["answer"][:500]

    # Twilio expects XML (TwiML) back
    twiml = f"""
    <Response>
        <Message>{answer}</Message>
    </Response>
    """
    return PlainTextResponse(content=twiml, media_type="application/xml")

