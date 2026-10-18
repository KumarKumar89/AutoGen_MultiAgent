# AutoGen Multi-Agent Ecosystem

This folder contains the original AutoGen tutorial artifacts and the Colab-ready HTML app.

## Interactive HTML app (recommended)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KumarKumar89/AutoGen_MultiAgent/blob/feature/colab-html-open-llm/AutoGen_MultiAgent/AutoGen_MultiAgent_Colab_HTML.ipynb)

The notebook starts a Gradio HTML form backed by open-weight Hugging Face models. It accepts pasted/uploaded CV and course catalogue inputs, displays role-based agent results, and creates a downloadable standalone HTML report. It requires no paid API key.

Select a GPU runtime if one is available. The first run downloads the model. The share link is public while active: only use redacted/demo CVs, never sensitive personal information.

## Files

- colab_html_app.py — Gradio UI, open-weight LLM inference, HTML preview/download.
- AutoGen_MultiAgent_Colab_HTML.ipynb — Google Colab launcher notebook.
- requirements-colab.txt — dependencies for the HTML app.
- run_live_multi_agent.py — existing command-line demo runner.
- fixed_multi_agent_ecosystem.py — original AutoGen/Groq workflow.
