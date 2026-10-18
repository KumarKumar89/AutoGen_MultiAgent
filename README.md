# Autogen Multi-Agent Ecosystem: Profile Analysis with Free LLMs

<img title="Logo" alt="Alt text" src="/imgs/cover_page.png">

## Description
This repository demonstrates how to build and execute a multi-agent system using **Autogen-style role-based agents** and free / open-weight LLMs. It covers profile/CV analysis, skill extraction, course recommendation, and a completion check.

**New (2026):** Interactive HTML app for Google Colab using open-weight Hugging Face models. No paid API key is required.

## Launch the interactive HTML app in Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KumarKumar89/AutoGen_MultiAgent/blob/feature/colab-html-open-llm/AutoGen_MultiAgent/AutoGen_MultiAgent_Colab_HTML.ipynb)

The notebook opens a form with fields for candidate label, target role, CV/profile text or file upload, course-catalogue text or file upload, and an open-weight model selector. It displays a formatted HTML report and offers the report as a downloadable standalone HTML file.

**Recommended runtime:** Google Colab → Runtime → Change runtime type → T4 GPU (if available). The default model is Qwen/Qwen2.5-0.5B-Instruct; alternatives include Qwen2.5 1.5B and TinyLlama 1.1B. The first use downloads model weights, so internet access is needed during setup. No paid model API key is used.

**Privacy:** The Gradio share link is publicly accessible while the notebook is running. Use redacted/demo CVs and do not submit sensitive personal information.

## Features

- **Role-based workflow:** CV Analyst, Skills Extractor, Courses Recommender, Resolution Checker.
- **HTML report:** Read the report directly in the notebook and download a self-contained HTML file.
- **Open-weight models:** Runs Hugging Face Transformers locally in the Colab runtime.
- **File inputs:** Paste text or upload .txt, .md, .csv, .pdf, or .docx files.
- **Catalogue grounding:** Course titles are shown as recommendations only when a valid model-selected ID maps to an entry in the supplied catalogue.
- **No API key path:** The interactive Colab flow does not need a paid inference API.
- **Original Groq / AutoGen notebooks:** Kept for comparison and experimentation.

## Quick start (interactive Colab app)

1. Open the Colab notebook using the badge above.
2. Run the cells from top to bottom.
3. Use the generated Gradio form to enter a redacted CV/profile and course catalogue.
4. Click **Analyze profile and generate HTML report**.
5. Review the report in the page or download the HTML file.

A GPU is recommended. CPU mode is supported by the default small model but may be slower. Model results are not a substitute for checking the candidate's claims or current course availability.

## Command-line zero-key demo

    cd AutoGen_MultiAgent
    # For the command-line runner, data/cv.txt and data/courses.txt must exist.
    python run_live_multi_agent.py

This existing command-line runner prints the conversation and writes live_output.json. Its built-in fallback contains predefined response logic; use the interactive Colab app above when you need real local model inference from entered CV text.

## Optional Ollama backend

    ollama pull llama3.2:1b
    ollama serve
    python run_live_multi_agent.py

## Original notebooks

| Notebook | Description |
|----------|-------------|
| AutoGen_MultiAgent_Colab_HTML.ipynb | Interactive HTML form using an open-weight LLM |
| AutoGen_MultiAgent_Colab.ipynb | Original zero-key tutorial runner |
| multi_agent_ecosistem.ipynb | Main profile analysis (Groq) |
| multi_agent_conversation.ipynb | Student–tutor conversation |
| multi_agent_stock_analysis.ipynb | Stock analysis group chat |

## Colab dependencies

    pip install -r requirements-colab.txt

The Colab runtime provides PyTorch; the requirements file intentionally does not overwrite the runtime-specific PyTorch/CUDA installation.

## Tests

Lightweight offline tests check HTML escaping and catalogue-ID grounding. They do not download model weights or prove that a specific Colab GPU is available. A live generation test requires network access to the model repository and enough runtime memory.

## License

See LICENSE.
