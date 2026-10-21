#!/usr/bin/env python3
"""
FastAPI service exposing the AutoGen Multi-Agent engine over HTML + JSON
endpoints so users can interact with the ecosystem remotely.

Endpoints
---------
GET  /                  interactive HTML chat/dashboard (no frontend build needed)
GET  /health            liveness + backend info
GET  /api/agents        list of agents + roles in the ecosystem
GET  /api/data          raw CV + courses catalogue (grounding data)
POST /api/run           run the full multi-agent pipeline on a query,
                        returns agent order, full chat history and final answer
GET  /docs              auto-generated OpenAPI (Swagger UI)

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
 or:  python app.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Reuse the multi-agent engine from the repository (no duplication of logic)
from run_live_multi_agent import (
    MockManager,
    try_ollama,
    read_txt,
    CV_PATH,
    COURSES_PATH,
    ROOT,
)

app = FastAPI(
    title="AutoGen Multi-Agent Ecosystem – Remote API",
    description=(
        "Profile-analysis multi-agent system (CV Analyser → Skills Extractor → "
        "Courses Recommender → Resolution Checker) exposed as remote HTML/JSON "
        "endpoints. Zero API keys required; optional Ollama backend."
    ),
    version="1.0.0",
)

START_TS = time.time()
RUN_COUNT = 0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    query: str = Field(
        default="",
        description="Free-text request. Empty = default 'Recommend courses for the profile'.",
    )
    max_turns: int = Field(default=30, ge=1, le=60, description="Max conversation turns.")
    use_ollama: bool = Field(default=True, description="Try local Ollama first, fall back to open-source engine.")


class ChatMessage(BaseModel):
    name: str
    role: str
    content: str
    tool_calls: Optional[List[dict]] = None


class RunResponse(BaseModel):
    mode: str
    query: str
    duration_s: float
    timestamp: str
    agent_order: List[str]
    history: List[ChatMessage]
    final: str


# ---------------------------------------------------------------------------
# HTML dashboard (single-file, self-contained, remotely interactive)
# ---------------------------------------------------------------------------
INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoGen Multi-Agent · Remote Console</title>
<style>
  :root{--bg:#0f1420;--card:#1a2233;--acc:#4f9cf9;--ok:#37c978;--txt:#e8edf5;--mut:#8b97ad}
  *{box-sizing:border-box}
  body{margin:0;font-family:ui-sans-serif,system-ui,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--txt)}
  header{padding:18px 28px;background:linear-gradient(90deg,#16203a,#0f1420);border-bottom:1px solid #26314a;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
  header h1{font-size:1.15rem;margin:0}
  header .pill{background:#223052;border:1px solid #34477a;color:var(--acc);padding:3px 10px;border-radius:999px;font-size:.75rem}
  main{max-width:1100px;margin:24px auto;padding:0 18px;display:grid;grid-template-columns:1fr 340px;gap:18px}
  @media(max-width:860px){main{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid #26314a;border-radius:12px;padding:16px}
  textarea{width:100%;min-height:74px;background:#121a2b;color:var(--txt);border:1px solid #2c3a58;border-radius:8px;padding:10px;font-size:.95rem;resize:vertical}
  button{background:var(--acc);color:#0b1220;border:0;border-radius:8px;padding:10px 18px;font-weight:700;cursor:pointer;margin-top:10px}
  button:disabled{opacity:.5;cursor:wait}
  .row{display:flex;gap:12px;align-items:center;margin-top:8px;flex-wrap:wrap}
  label.chk{color:var(--mut);font-size:.85rem;display:flex;gap:6px;align-items:center}
  #status{font-size:.85rem;color:var(--mut)}
  .msg{border-left:3px solid #34477a;margin:10px 0;padding:8px 12px;background:#121a2b;border-radius:0 8px 8px 0;white-space:pre-wrap;font-size:.9rem}
  .msg .who{font-weight:700;color:var(--acc)}
  .msg.tool{border-left-color:#b98a2b}.msg.tool .who{color:#e5b04a}
  .msg.final{border-left-color:var(--ok)}.msg.final .who{color:var(--ok)}
  .kv{display:flex;justify-content:space-between;font-size:.85rem;color:var(--mut);border-bottom:1px dashed #26314a;padding:5px 0}
  .kv b{color:var(--txt)}
  h2{font-size:.95rem;margin:4px 0 10px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em}
  a{color:var(--acc)}
  ul.agents{list-style:none;padding:0;margin:0;font-size:.88rem}
  ul.agents li{padding:6px 0;border-bottom:1px dashed #26314a}
  footer{text-align:center;color:#5a6789;font-size:.75rem;padding:18px}
</style>
</head>
<body>
<header>
  <h1>🤖 AutoGen Multi-Agent Ecosystem</h1>
  <span class="pill">Remote Console</span>
  <span class="pill">Zero API keys</span>
  <span class="pill"><a href="/docs" style="color:inherit;text-decoration:none">API Docs →</a></span>
</header>
<main>
  <section class="card">
    <h2>Run the agent ecosystem</h2>
    <textarea id="q" placeholder="e.g. Recommend courses for the profile">Recommend courses for the profile</textarea>
    <div class="row">
      <label class="chk"><input type="checkbox" id="ollama" checked> try local Ollama first</label>
      <button id="go" onclick="run()">▶ Run multi-agent pipeline</button>
      <span id="status"></span>
    </div>
    <div id="out"></div>
  </section>
  <aside>
    <div class="card">
      <h2>Pipeline KPIs</h2>
      <div class="kv"><span>Total runs</span><b id="k-runs">–</b></div>
      <div class="kv"><span>Last latency</span><b id="k-lat">–</b></div>
      <div class="kv"><span>Agents involved</span><b id="k-agents">–</b></div>
      <div class="kv"><span>Messages exchanged</span><b id="k-msgs">–</b></div>
      <div class="kv"><span>Backend mode</span><b id="k-mode">–</b></div>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>Agents</h2>
      <ul class="agents" id="agents">loading…</ul>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>Data flow</h2>
      <div style="font-size:.82rem;color:var(--mut);line-height:1.5">
        User query → <b style="color:var(--txt)">CV Analyser</b> (tool: read cv.txt)<br>
        → <b style="color:var(--txt)">Skills Extractor</b> (tool: read cv.txt)<br>
        → <b style="color:var(--txt)">Courses Recommender</b> (tool: read courses.txt)<br>
        → <b style="color:var(--txt)">Resolution Checker</b> → TERMINATE + final answer
      </div>
    </div>
  </aside>
</main>
<footer>AutoGen Multi-Agent · FastAPI remote console · /health · /api/run · /docs</footer>
<script>
async function loadAgents(){
  const r = await fetch('/api/agents'); const d = await r.json();
  document.getElementById('agents').innerHTML =
    d.agents.map(a=>`<li><b>${a.name}</b><br><span style="color:var(--mut)">${a.role}</span></li>`).join('');
}
async function run(){
  const btn=document.getElementById('go'), st=document.getElementById('status');
  btn.disabled=true; st.textContent='running agents…';
  const t0=performance.now();
  try{
    const res=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query:document.getElementById('q').value, use_ollama:document.getElementById('ollama').checked})});
    const d=await res.json();
    const lat=((performance.now()-t0)/1000).toFixed(2)+' s';
    document.getElementById('k-lat').textContent=lat;
    document.getElementById('k-runs').textContent=d.run_count;
    document.getElementById('k-agents').textContent=d.agent_order.length;
    document.getElementById('k-msgs').textContent=d.history.length;
    document.getElementById('k-mode').textContent=d.mode;
    let html='<h2>Chat history</h2>';
    d.history.forEach(m=>{
      const isTool=m.tool_calls&&m.tool_calls.length;
      const cls=m.name==='Resolution Checker Agent'&&/TERMINATE/.test(m.content)?'final':(isTool?'tool':'');
      html+=`<div class="msg ${cls}"><span class="who">${m.name}</span>`;
      if(isTool) html+=`<br><i style="color:#e5b04a">[TOOL CALL] ${m.tool_calls.map(t=>t.function.name+'('+t.function.arguments+')').join(', ')}</i>`;
      if(m.content) html+=`<br>${m.content.replace(/</g,'&lt;')}`;
      html+='</div>';
    });
    document.getElementById('out').innerHTML=html;
    st.textContent='done in '+lat;
  }catch(e){st.textContent='error: '+e}
  btn.disabled=false;
}
loadAgents();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    """Interactive HTML console for remote users."""
    return HTMLResponse(INDEX_HTML)


@app.get("/health")
def health():
    ollama = try_ollama("ping") is not None
    return {
        "status": "ok",
        "service": "autogen-multi-agent-api",
        "version": app.version,
        "uptime_s": round(time.time() - START_TS, 1),
        "runs": RUN_COUNT,
        "backend": {"ollama_available": ollama, "open_source_engine": True},
        "data_files": {
            "cv": Path(CV_PATH).exists(),
            "courses": Path(COURSES_PATH).exists(),
        },
    }


@app.get("/api/agents")
def agents():
    mgr = MockManager()
    roles = {
        "User": "UserProxy – executes file-reading tool calls on behalf of the agents",
        "CV Analyser": "Reads cv.txt via tool call and produces a structured profile analysis",
        "Skills Extractor": "Reads cv.txt via tool call and emits a comma-separated skill list",
        "Courses recommender": "Reads courses.txt + extracted skills and recommends a shortlist with rationale",
        "Resolution Checker Agent": "Verifies all specialists answered, aggregates the FINAL ANSWER and TERMINATEs",
    }
    return {
        "agents": [
            {"name": n, "role": roles.get(n, "")} for n in mgr.agents
        ],
        "speaker_selection": "hybrid: tool-call routing → reply-to-speaker → specialist chain → first unanswered",
    }


@app.get("/api/data")
def data():
    return {
        "cv_path": CV_PATH,
        "courses_path": COURSES_PATH,
        "cv": read_txt(CV_PATH),
        "courses": read_txt(COURSES_PATH),
    }


@app.post("/api/run", response_model=None)
def run_pipeline(req: RunRequest):
    """Execute the full multi-agent conversation and return everything."""
    global RUN_COUNT
    t0 = time.time()
    query = req.query.strip() or f"Recommend courses for the profile {CV_PATH}"

    history, mode = None, ""
    if req.use_ollama:
        h = try_ollama(query)
        if h:
            history, mode = h, "ollama-local"
    if history is None:
        history = MockManager().run(query, max_turns=req.max_turns)
        mode = "open-source-llama-style"

    RUN_COUNT += 1
    resp = {
        "mode": mode,
        "query": query,
        "duration_s": round(time.time() - t0, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_count": RUN_COUNT,
        "agent_order": [m["name"] for m in history],
        "history": [
            {**m, "content": m.get("content") or ""} for m in history
        ],
        "final": history[-1].get("content", ""),
    }
    # persist last run for forensics/debugging
    try:
        (ROOT / "last_api_run.json").write_text(
            __import__("json").dumps(resp, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass
    return JSONResponse(resp)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
