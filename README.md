# Autogen Multi-Agent Ecosystem: Profile Analysis with Free LLMs

<img title="Logo" alt="Project cover" src="AutoGen_MultiAgent/imgs/cover_page.png">

> **2026 update — now up & running out of the box:** one-command runner (`run.py`),
> a **FastAPI remote service with an interactive HTML console**, and a
> **forensic test-suite** (22 tests, all green). Zero API keys required.

---

## 1. Project objectives

| # | Objective | Status |
|---|-----------|--------|
| O1 | Demonstrate a working **multi-agent ecosystem** (CV Analyser → Skills Extractor → Courses Recommender → Resolution Checker) with hybrid speaker selection and tool integration | ✅ |
| O2 | Run with **zero paid API keys** — pure-Python open-source engine by default, optional real local Llama via Ollama, optional Groq path for the original notebooks | ✅ |
| O3 | Be **up & running with a single command** (`python run.py`) that self-heals assets and dependencies | ✅ new |
| O4 | **Expose the repository remotely** through HTML + JSON endpoints so any user can interact from a browser or API client (FastAPI) | ✅ new |
| O5 | Provide **forensic verification** of repo integrity, agent logic, data grounding and HTTP surface (`pytest`) | ✅ new |
| O6 | Keep the original tutorial notebooks (Groq/Autogen) available unchanged | ✅ |

---

## 2. Solution (high-level) architecture — SLD

```
                        ┌──────────────────────────────────────────────┐
  Remote users          │                 run.py  (Runner)             │
  (browser / curl /     │  python-check → asset-extraction → dep-check │
   SDK clients)         │  → launches CLI | server | test mode         │
        │               └───────────────┬──────────────────────────────┘
        ▼                               ▼
┌───────────────┐            ┌───────────────────────────┐
│  HTML console │  fetch()   │   FastAPI service (app.py)│
│  GET  /       │◄──────────►│  uvicorn  0.0.0.0:8000    │
└───────────────┘            │  /health /api/agents      │
                             │  /api/data  POST /api/run │
                             │  /docs (OpenAPI/Swagger)  │
                             └────────────┬──────────────┘
                                          │ imports (no code duplication)
                                          ▼
                             ┌───────────────────────────┐
                             │  Multi-Agent Engine        │
                             │  run_live_multi_agent.py   │
                             │ ─ MockManager (orchestr.)  │
                             │ ─ 5 agents + speaker policy│
                             │ ─ tool-calls (read_txt_*)  │
                             └───────┬───────────┬────────┘
                       backend probe │           │ file tools
                     (optional)      ▼           ▼
                             ┌────────────┐  ┌──────────────────┐
                             │  Ollama    │  │ Grounding data    │
                             │ llama3.2   │  │ data/cv.txt       │
                             │ :11434     │  │ data/courses.txt  │
                             └────────────┘  └──────────────────┘

  Legacy/tutorial layer: multi_agent_ecosistem.ipynb · multi_agent_conversation.ipynb
  multi_agent_stock_analysis.ipynb · fixed_multi_agent_ecosystem.py (Groq + autogen-agentchat)

  Verification layer:    tests/test_forensic.py (pytest, 22 checks)
```

**Layer responsibilities**

| Layer | Component | Responsibility |
|-------|-----------|----------------|
| Entry | `run.py` | Bootstrap: Python check, unzip `data/`+`imgs/`, install deps, dispatch cli/server/test |
| Presentation | `app.py` | FastAPI app: HTML console, REST endpoints, OpenAPI docs, KPI counters |
| Orchestration | `MockManager` in `run_live_multi_agent.py` | Speaker selection, turn loop, TERMINATE detection |
| Agents | 5 `MockAgent`s | CV analysis, skill extraction, course recommendation, resolution checking, tool execution |
| Data | `data/cv.txt`, `data/courses.txt` | Grounding corpus read through tool calls |
| Optional model | Ollama (`llama3.2:1b`) | Real local Llama when `http://127.0.0.1:11434` responds |
| QA | `tests/test_forensic.py` | Forensic regression of every layer above |

---

## 3. Data flow (one pipeline run)

```
User query ("Recommend courses for the profile")
   │
   ├─[if Ollama reachable]→ single-shot local Llama answer ─→ response (mode=ollama-local)
   │
   └─[else] Multi-agent conversation (mode=open-source-llama-style):
       1. CV Analyser            → emits tool_call read_txt_cv_analyse(cv.txt)
       2. User (UserProxy)       → executes the file tool, returns real CV text
       3. CV Analyser            → structured profile analysis (grounded on CV)
       4. Skills Extractor       → tool_call read_txt_cv_skills → comma-separated skills
       5. Courses recommender    → tool_call read_txt_courses → shortlist + rationale
       6. Resolution Checker     → CONTINUE until all specialists answered
       7. Resolution Checker     → aggregates FINAL ANSWER + TERMINATE
       8. Runner/service         → persists live_output.json / last_api_run.json
```

Every artefact returned to the client contains: `mode`, `query`, `duration_s`,
`agent_order`, full `history` (incl. tool calls) and `final`.

---

## 4. Code logic (key internals)

* **Agents as callables** — each `MockAgent.respond(history, last)` returns
  `(content, tool_calls)`. A non-empty `tool_calls` list means "I need data",
  exactly mirroring OpenAI-style function calling used by Autogen.
* **Hybrid speaker selection** (`MockManager.select`) — priority rules:
  1. pending `tool_calls` → route to **User** (UserProxy executes the file read);
  2. a User tool-result → return control to the **previous speaker**;
  3. a specialist finished → hand to **Resolution Checker Agent**;
  4. otherwise pick the **first specialist with no final answer yet**.
* **Termination** — the loop stops when the Resolution Checker emits a message
  containing `TERMINATE`, or after `max_turns` (default 30, API-capped at 60).
* **Grounding over hallucination** — recommendations are cross-checked against
  the literal titles present in `data/courses.txt` (see forensic test
  `test_grounding_not_hallucinated`).
* **Backend auto-detection** — `try_ollama()` probes `127.0.0.1:11434/api/tags`
  with a 2 s timeout; failure silently falls back to the zero-key engine, so the
  repo runs anywhere.
* **Single source of truth** — `app.py` imports the engine instead of
  duplicating it; the HTML console is served inline (no build step, no static
  files to ship).

---

## 5. KPIs & measurement

| KPI | Definition | Target | Measured by |
|-----|------------|--------|-------------|
| Pipeline completion | % of runs reaching `TERMINATE` with all 4 answers | 100 % | `tests::TestPipeline::test_terminates` |
| Specialist coverage | all specialists produced a final (non-tool) answer | 4 / 4 | `test_all_specialists_answered` |
| Tool-execution fidelity | declared tools actually executed by UserProxy | ≥ 2 tool reads/run | `test_tool_calls_were_executed` |
| Grounding accuracy | recommended courses exist verbatim in catalogue | 100 % | `test_grounding_not_hallucinated` |
| Turn efficiency | messages per completed run | ≤ 15 (typ. 12–14) | `agent_order` length in `/api/run` |
| API latency | p95 `/api/run` duration | < 1 s (offline engine) | `duration_s` field / dashboard |
| Availability | `/health` 200 with both data files present | ≥ 99 % | `tests::TestHTTPEndpoints::test_health` |
| Input safety | invalid payloads rejected | HTTP 422 | `test_invalid_payload_rejected` |
| Test health | forensic suite pass rate | 22 / 22 ✅ | `python run.py --test` |
| Time-to-first-run | fresh clone → interactive service | < 60 s, one command | `python run.py` |

The HTML console shows live values for total runs, last latency, agents
involved, messages exchanged and backend mode.

---

## 6. Quick start — up & running

### One command (recommended)
```bash
pip install -r AutoGen_MultiAgent/requirements.txt   # fastapi, uvicorn, pytest…
python run.py                # → http://localhost:8000  (interactive HTML console)
python run.py --cli          # run the pipeline once in the terminal
python run.py --test         # run the 22-test forensic suite
python run.py --port 9000    # custom port
```
`run.py` automatically extracts `data/` and `imgs/` from the bundled zips and
installs missing web dependencies.

### Manual equivalents
```bash
cd AutoGen_MultiAgent
python run_live_multi_agent.py                      # CLI engine
uvicorn app:app --host 0.0.0.0 --port 8000          # web service
```

### With a real local Llama (optional)
```bash
ollama pull llama3.2:1b && ollama serve             # engine auto-detects it
```

### Google Colab
Upload `AutoGen_MultiAgent/AutoGen_MultiAgent_Colab.ipynb` → Run all (no keys).

---

## 7. Remote interaction — exposed endpoints

| Method | Endpoint | Type | Purpose |
|--------|----------|------|---------|
| GET | `/` | **HTML** | Interactive chat console + live KPI panel (remote users run the ecosystem from the browser) |
| GET | `/health` | JSON | Liveness, uptime, run counter, backend availability, data-file check |
| GET | `/api/agents` | JSON | Agent roster, roles, speaker-selection policy |
| GET | `/api/data` | JSON | Raw grounding corpus (`cv.txt`, `courses.txt`) |
| POST | `/api/run` | JSON | Run the full multi-agent pipeline: `{query, max_turns, use_ollama}` → history + final |
| GET | `/docs` | HTML | Swagger UI (auto-generated OpenAPI) |

Example remote call:
```bash
curl -s localhost:8000/api/run -H 'Content-Type: application/json' \
     -d '{"query":"Recommend courses for the profile","use_ollama":false}' | jq .final
```

Last run is persisted to `AutoGen_MultiAgent/last_api_run.json` for forensics.

---

## 8. Repository layout

```
.
├── run.py                          ← one-command runner (server / cli / test)
├── README.md                       ← this document
├── LICENSE
├── tests/
│   └── test_forensic.py            ← 22-test forensic suite
└── AutoGen_MultiAgent/
    ├── app.py                      ← FastAPI service + HTML console
    ├── run_live_multi_agent.py     ← multi-agent engine (zero-key + Ollama)
    ├── requirements.txt            ← web + engine + test dependencies
    ├── data/                       ← cv.txt, courses.txt (extracted from zip)
    ├── imgs/                       ← cover_page.png
    ├── AutoGen_MultiAgent_Colab.ipynb
    ├── fixed_multi_agent_ecosystem.py
    └── multi_agent_*.ipynb         ← original Groq/Autogen tutorials
```

### Original notebooks (unchanged)
| Notebook | Description |
|----------|-------------|
| `multi_agent_ecosistem.ipynb` | Main multi-agent profile analysis (Groq) |
| `multi_agent_conversation.ipynb` | Simple student–tutor conversation |
| `multi_agent_stock_analysis.ipynb` | Stock analysis group chat |

Groq path still available: set `GROQ_API_KEY=gsk_...` in `.env`.

---

## 9. Forensic test results (current run)

```
$ python run.py --test
======================== 22 passed in 0.91s ========================
  TestRepoIntegrity ..... 4 passed   (files, assets, grounding content)
  TestEngineLogic ....... 6 passed   (roster, routing, tools, errors)
  TestPipeline .......... 5 passed   (TERMINATE, coverage, anti-hallucination)
  TestHTTPEndpoints ..... 7 passed   (HTML, health, agents, data, run, docs, 422)
```

## 10. License
See [LICENSE](LICENSE).
