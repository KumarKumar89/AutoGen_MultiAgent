# Autogen Multi-Agent Ecosystem: Profile Analysis with Free LLMs

<img title="Logo" alt="Alt text" src="/imgs/cover_page.png">


## Description
This repository demonstrates how to build and execute a multi-agent system using **Autogen** and free LLMs with **Groq**. It serves as a step-by-step tutorial to help users understand the basics of multi-agent systems, tool integration, and hybrid agent selection methods.

---

## Features
- **Multi-Agent Setup**: Includes agents for analyzing CVs, extracting skills, recommending courses, and verifying task completion.
- **Hybrid Agent Selection**: Implements a custom method for dynamically selecting the next agent based on the task flow.
- **Tool Integration**: Demonstrates how to use external tools for processing tasks.
- **Custom Configuration**: Allows experimentation with LLM configurations like `llama31` and `gemma2` for optimized performance.

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

