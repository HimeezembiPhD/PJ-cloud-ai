from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

PJ_SYSTEM_PROMPT = """
You are PJ, a helpful personal companion AI.
Style: warm, concise, and practical.
If you don’t know something, say so and ask a short follow-up question.
"""

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"assistant": "PJ", "status": "online"}

@app.post("/chat")
def chat(req: ChatRequest):
    # Temporary “rule-based PJ” so it feels less like echo (until we add an LLM)
    msg = req.message.strip()

    if not msg:
        return {"assistant": "PJ", "reply": "Say something and I’m here 🙂"}

    if "hello" in msg.lower() or "hi" in msg.lower():
        return {"assistant": "PJ", "reply": "Hey 🙂 What do you want to do today?"}

    return {"assistant": "PJ", "reply": f"I hear you: {msg}. Want me to help you plan, learn, or build something?"}
