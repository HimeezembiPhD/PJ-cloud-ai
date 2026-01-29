from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import Dict, List, Literal

from openai import OpenAI

app = FastAPI(title="PJ Cloud AI", version="1.1.0")

PJ_SYSTEM_PROMPT = """
You are PJ, a helpful personal companion AI.
Style: warm, concise, and practical.

Rules:
- If you don’t know something, say so and ask a short follow-up question.
- You may share publicly available, official resources (government sites, public registries, official associations).
- Do help with illegal sourcing or purchasing controlled substances.
- If a request is about medical or legal resources, provide safe guidance and official links instead of refusing completely.
""".strip()

Role = Literal["system", "user", "assistant"]

# In-memory store: resets if Render restarts (good for MVP)
SESSIONS: Dict[str, List[dict]] = {}
MAX_TURNS = 20  # keep last 20 user+assistant turns to control token usage


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/")
def root():
    return {"assistant": "PJ", "status": "online"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatRequest):
    msg = (req.message or "").strip()
    session_id = (req.session_id or "").strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    if not msg:
        return {"assistant": "PJ", "reply": "Say something and I’m here 🙂", "session_id": session_id}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing in Render env vars")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    # Initialize session history if new
    if session_id not in SESSIONS:
        SESSIONS[session_id] = [{"role": "system", "content": PJ_SYSTEM_PROMPT}]

    history = SESSIONS[session_id]

    # Append user message
    history.append({"role": "user", "content": msg})

    # Trim history (keep system + last MAX_TURNS*2 messages)
    # Each turn usually has 2 messages: user + assistant
    system_msg = history[0]
    recent = history[1:][-MAX_TURNS * 2 :]
    history = [system_msg] + recent
    SESSIONS[session_id] = history

    try:
        response = client.chat.completions.create(
            model=model,
            messages=history,
            temperature=0.7,
        )
        reply = response.choices[0].message.content or ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")

    # Append assistant reply
    SESSIONS[session_id].append({"role": "assistant", "content": reply})

    return {"assistant": "PJ", "reply": reply, "session_id": session_id}
