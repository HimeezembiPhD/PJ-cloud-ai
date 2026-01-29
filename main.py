from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from openai import OpenAI

app = FastAPI(title="PJ Cloud AI", version="1.0.0")

PJ_SYSTEM_PROMPT = """
You are PJ, a helpful personal companion AI.
Style: warm, concise, and practical.
If you don’t know something, say so and ask a short follow-up question.
""".strip()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"assistant": "PJ", "status": "online"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/check-openai")
def check_openai():
    """
    Useful to verify OPENAI_API_KEY exists on Render without exposing the key.
    """
    return {
        "openai_key_loaded": bool(os.getenv("OPENAI_API_KEY")),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }


@app.post("/chat")
def chat(req: ChatRequest):
    msg = (req.message or "").strip()

    if not msg:
        return {"assistant": "PJ", "reply": "Say something and I’m here 🙂"}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY missing. Add it in Render Environment Variables."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PJ_SYSTEM_PROMPT},
                {"role": "user", "content": msg},
            ],
            temperature=0.7,
        )

        reply = response.choices[0].message.content
        return {"assistant": "PJ", "reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")
