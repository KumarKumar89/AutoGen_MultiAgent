# Autogen Multi-Agent Ecosystem: Profile Analysis with Free LLMs

<img title="Logo" alt="Alt text" src="/imgs/cover_page.png">


## Description
This repository demonstrates how to build and execute a multi-agent system using **Autogen** and free LLMs with **Groq**. It serves as a step-by-step tutorial to help users understand the basics of multi-agent systems, tool integration, and hybrid agent selection methods.

---

## Features
- **Multi-Agent Setup**: Includes agents for analyzing CVs, extracting skills, recommending courses, and verifying task completion.
- **Hybrid Agent Selection**: Implements a custom method for dynamically selecting the next agent based on the task flow.
- **Tool Integration**: Demonstrates how to use external tools for processing tasks.
- **Custom Configuration**: Allows experimentation with LLM configurations like `llama31` and 'groq' / `gemma2` for optimized performance.

---

## Requirements
- Python used version: **3.10.15**
- Dependencies listed in `requirements.txt`
- API Key for **Groq**
- `.env` file with the following variable:
  ```plaintext
  GROQ_API_KEY=<your_api_key>
    ```

```python
pip install -r requirements.txt
```


IMPORTANT INSTRUCTION:-  If some file is not working in the python environment try to use google colab

---

## Web service (2026) — remote interaction

The repository is now exposed remotely via a FastAPI service with an interactive HTML console.

```bash
# from the repo root (auto-checks deps & extracts data assets):
python ../run.py                # → http://localhost:8000  (HTML console + KPI panel)
python ../run.py --cli          # one-shot pipeline in terminal
python ../run.py --test         # forensic test-suite (22 tests)

# or directly:
uvicorn app:app --host 0.0.0.0 --port 8000
```

| Endpoint | Purpose |
|--------|---------|
| `GET /` | Interactive HTML chat console (runs the multi-agent pipeline remotely) |
| `GET /health` | Liveness, uptime, backend (Ollama) probe, data-file check |
| `GET /api/agents` | Agent roster + speaker-selection policy |
| `GET /api/data` | Raw grounding corpus (`cv.txt`, `courses.txt`) |
| `POST /api/run` | Run pipeline: `{query, max_turns, use_ollama}` → history + final |
| `GET /docs` | Swagger UI (OpenAPI) |

See the root [README.md](../README.md) for the SLD architecture, data flow, code logic and KPIs.
