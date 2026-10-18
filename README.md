# Autogen Multi-Agent Ecosystem: Profile Analysis with Free LLMs

<img title="Logo" alt="Alt text" src="/imgs/cover_page.png">

## Description
This repository demonstrates how to build and execute a multi-agent system using **Autogen** and free / open-source LLMs.  
It is a step-by-step tutorial covering multi-agent systems, tool integration, and hybrid agent selection.

**New (2026):** Zero-API-key open-source path + Colab notebook + optional local Llama (Ollama).

---

## Features
- **Multi-Agent Setup**: CV Analyser, Skills Extractor, Courses Recommender, Resolution Checker
- **Hybrid Agent Selection**: Custom speaker-selection logic
- **Tool Integration**: File-reading tools executed by UserProxy
- **Zero API keys mode**: Pure-Python Llama-style engine that works offline
- **Optional real local Llama**: Automatically uses Ollama when available
- **Google Colab ready**: `AutoGen_MultiAgent_Colab.ipynb` runs without any paid key
- **Original Groq path**: Still available when `GROQ_API_KEY` is set

---

## Quick start (zero API keys)

```bash
cd AutoGen_MultiAgent
# data/ must contain cv.txt and courses.txt (extract from the supplied zip if needed)
python run_live_multi_agent.py
```

This runs the improved open-source multi-agent engine and prints the full conversation + final course recommendations.

### With real local Llama (Ollama)
```bash
ollama pull llama3.2:1b   # or llama3.1:8b if you have more RAM
ollama serve
python run_live_multi_agent.py
```

### Google Colab
1. Upload `AutoGen_MultiAgent/AutoGen_MultiAgent_Colab.ipynb`
2. Runtime → Run all  
   (creates the data files and runs the full multi-agent flow with no keys)

---

## Original notebooks
| Notebook | Description |
|----------|-------------|
| `multi_agent_ecosistem.ipynb` | Main multi-agent profile analysis (Groq) |
| `multi_agent_conversation.ipynb` | Simple student–tutor conversation |
| `multi_agent_stock_analysis.ipynb` | Stock analysis group chat |

### Requirements (original Groq path)
- Python 3.10+
- `pip install -r requirements.txt`
- Free Groq API key → put in `.env`:
  ```
  GROQ_API_KEY=gsk_...
  ```

```bash
pip install -r requirements.txt
```

---

## Files added / upgraded
- `run_live_multi_agent.py` – zero-key open-source engine + Ollama support
- `AutoGen_MultiAgent_Colab.ipynb` – Colab-ready notebook (no keys)
- `fixed_multi_agent_ecosystem.py` – cleaned Autogen script (dotenv, non-Colab)

---

## License
See LICENSE.
