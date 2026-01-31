from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import time
from typing import Dict, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup
from urllib.parse import quote

from openai import OpenAI

app = FastAPI(title="PJ Cloud AI", version="1.6.0")

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

# ---- sessions (in-memory MVP) ----
SESSIONS: Dict[str, List[dict]] = {}
SESSION_META: Dict[str, float] = {}  # session_id -> last_seen_epoch

MAX_TURNS = 20
DEFAULT_SEARCH_LIMIT = 5
SESSION_TTL_SECONDS = int(os.getenv("PJ_SESSION_TTL_SECONDS", str(6 * 60 * 60)))
CLEANUP_INTERVAL_SECONDS = 60
_last_cleanup = 0.0


class ChatRequest(BaseModel):
    session_id: str
    message: str


def _cleanup_sessions_if_needed() -> None:
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < CLEANUP_INTERVAL_SECONDS:
        return
    _last_cleanup = now

    dead = [sid for sid, last_seen in SESSION_META.items() if now - last_seen > SESSION_TTL_SECONDS]
    for sid in dead:
        SESSION_META.pop(sid, None)
        SESSIONS.pop(sid, None)


def current_time_for(place: str) -> Optional[str]:
    key = place.strip().lower()
    tz = CITY_TZ.get(key)
    if not tz:
        return None
    try:
        now = datetime.now(ZoneInfo(tz))
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return None


def extract_place_for_time_question(text: str) -> Optional[str]:
    t = text.lower().strip()
    patterns = ["current time in", "time in", "what time is it in"]
    for p in patterns:
        if p in t:
            place = t.split(p, 1)[1].strip().strip(" ?!.,")
            if "," in place:
                place = place.split(",", 1)[0].strip()

            aliases = {
                "nyc": "new york",
                "new york city": "new york",
                "la": "los angeles",
                "l.a.": "los angeles",
            }
            if place in aliases:
                return aliases[place]

            parts = place.split()
            for n in (3, 2, 1):
                if len(parts) >= n:
                    cand = " ".join(parts[:n])
                    if cand in CITY_TZ:
                        return cand
            return place
    return None


def should_search_web(user_msg: str) -> bool:
    text = user_msg.lower()
    triggers = [
        "search", "find", "look up", "lookup", "where can i",
        "official website", "official link", "directory", "providers",
        "clinic", "contact", "phone number", "address", "near me",
        "job", "jobs", "hiring", "vacancy", "vacancies", "stellenangebot",
        "stellenangebote", "reinigungskraft", "gebäudereinigung", "bewerben"
    ]
    return any(t in text for t in triggers)


async def ddg_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> List[dict]:
    q = query.strip()
    if not q:
        return []
    url = f"https://duckduckgo.com/html/?q={quote(q)}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PJCloudAI/1.0)"}

    async with httpx.AsyncClient(timeout=20.0, headers=headers, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")  # no lxml dependency
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


@app.get("/")
async def root():
    return {"assistant": "PJ", "status": "online"}


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/search")
async def search(q: str, limit: int = DEFAULT_SEARCH_LIMIT):
    _cleanup_sessions_if_needed()

    if not q.strip():
        raise HTTPException(status_code=400, detail="q is required")
    try:
        limit_i = int(limit)
    except Exception:
        limit_i = DEFAULT_SEARCH_LIMIT
    limit_i = max(1, min(limit_i, 10))

    try:
        results = await ddg_search(q, limit=limit_i)
        return {"query": q, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/chat")
async def chat(req: ChatRequest):
    _cleanup_sessions_if_needed()

    msg = (req.message or "").strip()
    session_id = (req.session_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    SESSION_META[session_id] = time.time()

    if not msg:
        return {"assistant": "PJ", "reply": "Say something and I’m here 🙂", "session_id": session_id}

    # World-clock direct handling
    place = extract_place_for_time_question(msg)
    if place:
        t = current_time_for(place)
        if t:
            return {"assistant": "PJ", "reply": f"The current time in {place.title()} is: {t}", "session_id": session_id}
        known = ", ".join(sorted(CITY_TZ.keys())[:12]) + " ..."
        return {"assistant": "PJ", "reply": f"I don’t know the timezone for '{place}'. Try: {known}", "session_id": session_id}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing in env vars")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    if session_id not in SESSIONS:
        SESSIONS[session_id] = [{"role": "system", "content": PJ_SYSTEM_PROMPT}]
    history = SESSIONS[session_id]

    # best-effort web context (do NOT store)
    web_context: Optional[str] = None
    if should_search_web(msg):
        try:
            results = await ddg_search(msg, limit=5)
            if results:
                lines = [f"- {r['title']} — {r['url']}" for r in results]
                web_context = "Web search results (use these links; do not invent links):\n" + "\n".join(lines)
        except Exception:
            web_context = None

    history.append({"role": "user", "content": msg})

    # trim stored conversation only
    system_msg = history[0]
    recent = history[1:][-MAX_TURNS * 2:]
    history = [system_msg] + recent
    SESSIONS[session_id] = history

    messages_for_request = list(history)
    if web_context:
        messages_for_request.append({"role": "system", "content": web_context})

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages_for_request,
            temperature=0.7,
        )
        reply = resp.choices[0].message.content or ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")

    SESSIONS[session_id].append({"role": "assistant", "content": reply})
    SESSION_META[session_id] = time.time()

    return {"assistant": "PJ", "reply": reply, "session_id": session_id}


# -------------------- UI (Text + Voice where supported) --------------------

CHAT_UI_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
  <title>PJ</title>
  <style>
    :root{color-scheme:dark}
    body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;background:#0b0f19;color:#e8e8e8}
    .wrap{max-width:880px;margin:0 auto;padding:16px}
    .top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:12px 0}
    .brand{display:flex;align-items:center;gap:10px}
    .dot{width:10px;height:10px;border-radius:99px;background:#16a34a;box-shadow:0 0 0 4px rgba(22,163,74,.15)}
    .title{font-weight:700;font-size:20px}
    .muted{opacity:.75;font-size:12px}
    .card{background:#121a2a;border:1px solid #1f2a44;border-radius:18px;overflow:hidden}
    .msgs{height:65vh;overflow:auto;padding:14px;background:#0f1626;scroll-behavior:smooth}
    .row{display:flex;gap:10px;padding:12px;border-top:1px solid #1f2a44;background:#121a2a}
    input{flex:1;padding:12px 12px;border-radius:14px;border:1px solid #24314f;background:#0f1626;color:#fff;outline:none}
    button{padding:12px 14px;border-radius:14px;border:0;background:#2a64ff;color:#fff;cursor:pointer;font-weight:600}
    button.secondary{background:#1b2438;border:1px solid #24314f}
    button:disabled{opacity:.6;cursor:not-allowed}
    .msg{margin:10px 0;display:flex}
    .me{justify-content:flex-end}
    .pj{justify-content:flex-start}
    .bubble{max-width:78%;padding:10px 12px;border-radius:16px;line-height:1.4;white-space:pre-wrap;word-wrap:break-word}
    .me .bubble{background:#2a64ff}
    .pj .bubble{background:#1b2438}
    a{color:#9bb7ff}
    .typing{opacity:.7;font-size:13px;margin:6px 0 0 2px}
    .pill{padding:4px 8px;border:1px solid #24314f;border-radius:999px;font-size:12px;opacity:.85}
    .banner{margin:10px 0;padding:10px 12px;border:1px solid #24314f;background:#0f1626;border-radius:14px;font-size:13px;opacity:.9;display:none}
    .banner strong{font-weight:700}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div class="brand">
        <div class="dot"></div>
        <div>
          <div class="title">PJ</div>
          <div class="muted">Text everywhere • Voice where supported</div>
        </div>
      </div>
      <div class="pill">Session: <span id="sid"></span></div>
    </div>

    <div id="banner" class="banner"></div>

    <div class="card">
      <div id="msgs" class="msgs"></div>
      <div class="row">
        <button id="mic" class="secondary" title="Talk">🎤</button>
        <input id="text" placeholder="Message PJ…" autocomplete="off" inputmode="text" />
        <button id="send">Send</button>
        <button id="speak" class="secondary" title="PJ voice on/off">🔊 On</button>
        <button id="clear" class="secondary" title="Clear chat">Clear</button>
      </div>
    </div>

    <div id="typing" class="typing" style="display:none;">PJ is typing…</div>
  </div>

<script>
  const msgs = document.getElementById("msgs");
  const text = document.getElementById("text");
  const send = document.getElementById("send");
  const clearBtn = document.getElementById("clear");
  const typing = document.getElementById("typing");
  const banner = document.getElementById("banner");
  const micBtn = document.getElementById("mic");

  // --- Session id ---
  let sessionId = localStorage.getItem("pj_session_id");
  if(!sessionId){
    sessionId = "s_" + Math.random().toString(36).slice(2);
    localStorage.setItem("pj_session_id", sessionId);
  }
  document.getElementById("sid").textContent = sessionId;

  let history = JSON.parse(localStorage.getItem("pj_ui_history") || "[]");

  // --- XSS-safe: escape first, then linkify ---
  function escapeHtml(str){
    return String(str)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }
  function linkify(escapedStr){
    return escapedStr.replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
  }

  function addBubble(who, content){
    const div = document.createElement("div");
    div.className = "msg " + who;

    const b = document.createElement("div");
    b.className = "bubble";

    const safe = linkify(escapeHtml(content)).replace(/\n/g, "<br/>");
    b.innerHTML = safe;

    div.appendChild(b);
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;

    history.push({ who, content });
    localStorage.setItem("pj_ui_history", JSON.stringify(history.slice(-200)));
  }

  function renderSaved(){
    msgs.innerHTML = "";
    if(history.length === 0){
      addBubble("pj", "Hey 🙂 I’m PJ. Talk or type to start.");
      return;
    }
    // render without re-saving
    const saved = history.slice();
    history = [];
    for(const m of saved){
      addBubble(m.who, m.content);
    }
  }

  // ---- Voice Out (TTS): widely supported, but voices vary ----
  let voiceOn = localStorage.getItem("pj_voice_on") !== "false";
  const speakBtn = document.getElementById("speak");
  function updateSpeakBtn(){ speakBtn.textContent = voiceOn ? "🔊 On" : "🔇 Off"; }
  updateSpeakBtn();

  speakBtn.onclick = () => {
    voiceOn = !voiceOn;
    localStorage.setItem("pj_voice_on", String(voiceOn));
    if(!voiceOn && window.speechSynthesis) window.speechSynthesis.cancel();
    updateSpeakBtn();
  };

  function speak(textToSpeak){
    if(!voiceOn) return;
    if(!window.speechSynthesis) return;
    const u = new SpeechSynthesisUtterance(String(textToSpeak || ""));
    u.rate = 1.0;
    u.pitch = 1.0;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  }

  // ---- Voice In (STT): NOT universal. Use feature-detection + graceful fallback ----
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let rec = null;
  let recognizing = false;

  function showBanner(msg){
    banner.style.display = "block";
    banner.innerHTML = msg;
  }

  const isSecure = location.protocol === "https:" || location.hostname === "localhost";

  if (!SpeechRecognition) {
    // Firefox/Safari typically: no STT
    micBtn.disabled = true;
    micBtn.title = "Voice input not supported in this browser";
    showBanner("<strong>Note:</strong> Voice input isn’t supported in this browser. Text chat works everywhere.");
  } else if (!isSecure) {
    // STT requires HTTPS in most browsers
    micBtn.disabled = true;
    micBtn.title = "Voice input requires HTTPS";
    showBanner("<strong>Note:</strong> Voice input needs <strong>HTTPS</strong>. Open this site via HTTPS to enable the mic.");
  } else {
    rec = new SpeechRecognition();
    // Prefer the browser language; user can change it later if needed
    rec.lang = navigator.language || "en-US";
    rec.interimResults = true;
    rec.continuous = false;

    rec.onstart = () => {
      recognizing = true;
      micBtn.textContent = "⏺️";
      micBtn.classList.remove("secondary");
    };

    rec.onend = () => {
      recognizing = false;
      micBtn.textContent = "🎤";
      micBtn.classList.add("secondary");
    };

    rec.onerror = (e) => {
      addBubble("pj", "Mic error: " + (e.error || "unknown") + " (try Chrome/Edge)");
    };

    rec.onresult = (event) => {
      let transcript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      text.value = transcript.trim();
    };

    micBtn.onclick = () => {
      try{
        if (!recognizing) rec.start();
        else rec.stop();
      }catch(e){
        addBubble("pj", "Mic failed to start. Try reloading the page.");
      }
    };

    rec.addEventListener("end", () => {
      const val = text.value.trim();
      if (val) sendMsg();
    });
  }

  async function sendMsg(){
    const m = text.value.trim();
    if(!m) return;

    text.value = "";
    addBubble("me", m);

    send.disabled = true;
    typing.style.display = "block";

    try{
      const res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ session_id: sessionId, message: m })
      });

      const data = await res.json();
      if(!res.ok){
        addBubble("pj", "Error: " + (data.detail || res.statusText));
      }else{
        addBubble("pj", data.reply || "(no reply)");
        speak(data.reply || "");
      }
    }catch(e){
      addBubble("pj", "Network error: " + e.message);
    }finally{
      typing.style.display = "none";
      send.disabled = false;
      text.focus();
    }
  }

  send.onclick = sendMsg;

  // Better mobile behavior: Enter sends, Shift+Enter newline
  text.addEventListener("keydown", (e) => {
    if(e.key === "Enter" && !e.shiftKey){
      e.preventDefault();
      sendMsg();
    }
  });

  clearBtn.onclick = () => {
    history = [];
    localStorage.removeItem("pj_ui_history");
    msgs.innerHTML = "";
    addBubble("pj", "Chat cleared. Talk or type to start again.");
  };

  renderSaved();
  text.focus();
</script>
</body>
</html>
"""

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return CHAT_UI_HTML
