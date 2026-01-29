from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import Dict, List, Literal, Optional

from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup
from urllib.parse import quote

from openai import OpenAI

app = FastAPI(title="PJ Cloud AI", version="1.3.0")

PJ_SYSTEM_PROMPT = """
You are PJ, a helpful personal companion AI.
Style: warm, concise, and practical.

Rules:
- If you don’t know something, say so and ask a short follow-up question.
- Do NOT claim you can browse the internet unless the server provides search results.
- You may share publicly available, official resources (government sites, public registries, official associations).
- Do NOT help with illegal sourcing or purchasing controlled substances.
- If a request is about medical or legal resources, provide safe guidance and official links instead of refusing completely.
""".strip()

# ✅ City -> timezone (IANA timezones)
CITY_TZ = {
    "windhoek": "Africa/Windhoek",
    "london": "Europe/London",
    "berlin": "Europe/Berlin",
    "paris": "Europe/Paris",
    "rome": "Europe/Rome",
    "madrid": "Europe/Madrid",
    "lisbon": "Europe/Lisbon",
    "new york": "America/New_York",
    "los angeles": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "toronto": "America/Toronto",
    "dubai": "Asia/Dubai",
    "delhi": "Asia/Kolkata",
    "tokyo": "Asia/Tokyo",
    "singapore": "Asia/Singapore",
    "sydney": "Australia/Sydney",
}

def current_time_for(place: str) -> Optional[str]:
    key = place.strip().lower()
    tz = CITY_TZ.get(key)
    if not tz:
        return None
    now = datetime.now(ZoneInfo(tz))
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


Role = Literal["system", "user", "assistant"]

# In-memory store: resets if Render restarts (good for MVP)
SESSIONS: Dict[str, List[dict]] = {}
MAX_TURNS = 20  # keep last 20 user+assistant turns to control token usage

DEFAULT_SEARCH_LIMIT = 5


class ChatRequest(BaseModel):
    session_id: str
    message: str


def ddg_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> List[dict]:
    """
    Simple web search using DuckDuckGo HTML results.
    Returns list of {title, url}.
    """
    q = query.strip()
    if not q:
        return []

    url = f"https://duckduckgo.com/html/?q={quote(q)}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PJCloudAI/1.0)"}

    with httpx.Client(timeout=20.0, headers=headers, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    results: List[dict] = []

    for a in soup.select("a.result__a"):
        title = a.get_text(strip=True)
        href = a.get("href")
        if not title or not href:
            continue
        results.append({"title": title, "url": href})
        if len(results) >= limit:
            break

    return results


def should_search_web(user_msg: str) -> bool:
    """
    Lightweight heuristic: triggers search when user asks to find/search/lookup/providers/etc.
    """
    text = user_msg.lower()
    triggers = [
        "search", "find", "look up", "lookup", "where can i",
        "official website", "official link", "directory", "providers",
        "clinic", "contact", "phone number", "address", "near me",
    "job", "jobs", "hiring", "vacancy", "vacancies", "stellenangebot",
    "stellenangebote", "reinigungskraft", "gebäudereinigung", "bewerben"
    ]
    return any(t in text for t in triggers)


def extract_place_for_time_question(text: str) -> Optional[str]:
    """
    Detect: "time in london", "current time in tokyo", "what time is it in berlin"
    Returns the extracted place string if found.
    """
    t = text.lower().strip()

    # Patterns we support
    patterns = ["current time in", "time in", "what time is it in"]

    for p in patterns:
        if p in t:
            place = t.split(p, 1)[1].strip()
            place = place.strip(" ?!.,")

            # handle inputs like "london uk" (we take first word unless it's "new york")
            if place.startswith("new york"):
                return "new york"

            # take first part before comma
            if "," in place:
                place = place.split(",", 1)[0].strip()

            # if multiple words, keep 2 words max (for city names)
            parts = place.split()
            if len(parts) >= 2:
                # allow 2-word cities like "los angeles"
                two = " ".join(parts[:2])
                if two in CITY_TZ:
                    return two
                return parts[0]

            return place

    return None


@app.get("/")
def root():
    return {"assistant": "PJ", "status": "online"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/search")
def search(q: str, limit: int = DEFAULT_SEARCH_LIMIT):
    """
    Test endpoint: GET /search?q=...&limit=5
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="q is required")

    limit = max(1, min(int(limit), 10))

    try:
        results = ddg_search(q, limit=limit)
        return {"query": q, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/chat")
def chat(req: ChatRequest):
    msg = (req.message or "").strip()
    session_id = (req.session_id or "").strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    if not msg:
        return {"assistant": "PJ", "reply": "Say something and I’m here 🙂", "session_id": session_id}

    # ✅ Handle world-clock questions directly (no OpenAI required)
    place = extract_place_for_time_question(msg)
    if place:
        t = current_time_for(place)
        if t:
            return {
                "assistant": "PJ",
                "reply": f"The current time in **{place.title()}** is: {t}",
                "session_id": session_id
            }
        else:
            known = ", ".join(sorted(CITY_TZ.keys())[:12]) + " ..."
            return {
                "assistant": "PJ",
                "reply": (
                    f"I don’t know the timezone for '{place}'.\n\n"
                    f"Try one of these: {known}\n"
                    f"Or tell me the timezone name like: `Europe/London`."
                ),
                "session_id": session_id
            }

    # ✅ OpenAI settings
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing in Render env vars")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    # Initialize session history if new
    if session_id not in SESSIONS:
        SESSIONS[session_id] = [{"role": "system", "content": PJ_SYSTEM_PROMPT}]

    history = SESSIONS[session_id]

    # Optional: fetch web results and inject into context
    web_context: Optional[str] = None
    if should_search_web(msg):
        try:
            results = ddg_search(msg, limit=5)
            if results:
                lines = [f"- {r['title']} — {r['url']}" for r in results]
                web_context = "Web search results (use these links; do not invent links):\n" + "\n".join(lines)
        except Exception:
            web_context = None

    # Append user message
    history.append({"role": "user", "content": msg})

    # If we have web results, add them
    if web_context:
        history.append({"role": "system", "content": web_context})

    # Trim history (keep system + last MAX_TURNS*2 messages)
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


