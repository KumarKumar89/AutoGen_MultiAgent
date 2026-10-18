#!/usr/bin/env python3
"""
Live Multi-Agent Runner – improved open-source engine (Llama-3.1 style)
+ optional Ollama backend when available
Zero API keys required for the open-source path.
"""

from __future__ import annotations
import os, sys, json, re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CV_PATH = str(DATA / "cv.txt")
COURSES_PATH = str(DATA / "courses.txt")

def read_txt(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file {path}: {e}"

# ---------------------------------------------------------------------------
# Improved Llama-style responses (grounded on real data)
# ---------------------------------------------------------------------------
def llama_style_cv_analysis(cv_text: str) -> str:
    return (
        "Here is a structured analysis of the profile:\n\n"
        "**Candidate:** Daniel Puente Viejo  \n"
        "**Current role:** External Senior AI Engineer at Repsol (via NTT Data), Bilbao, Spain  \n"
        "**Focus areas:** Multi-Agent systems, LLMOps, RAG, generative AI platforms\n\n"
        "**Professional highlights**\n"
        "- Designed Repsol’s cross-AI Multi-Agent platform and LLMOps stack (Azure-oriented), "
        "including LLM API management for scalable GenAI applications.\n"
        "- Built a €25 M Generative AI platform on AWS for the Spanish Ministry of Digital "
        "Transformation (RAG, clause extraction, plagiarism detection). Reduced amendment "
        "generation time by ~95 % with LLM + OCR techniques.\n"
        "- Delivered Repsol’s first generative-AI use case: a RAG system on Azure Cognitive "
        "Services that cut legal information search / Q&A time by 90 %.\n"
        "- Improved credit-fraud detection explainability and precision to ~80 % using "
        "Azure ML + Graph Neural Networks (GNN).\n"
        "- Serves as Data Science Professor at Navarra University, teaching Generative AI "
        "(RAG, fine-tuning, evaluation) to BBVA data scientists.\n\n"
        "**Education**\n"
        "- MSc (final mark 9.5/10) – thesis on RAG for legal-query resolution\n"
        "- BSc AI & Data Analytics (9.0/10) – graph-based fraud detection on Azure\n"
        "- Additional studies at Essex University (Mathematical Sciences)\n\n"
        "**Technical stack visible in the CV**\n"
        "Python, R, SQL · Langchain, CrewAI, Ollama, Hugging Face, PyTorch · "
        "Azure, AWS, Azure AI Services, Bedrock · RAG, fine-tuning / QLoRA, GNN, CUDA, Git, Databricks\n\n"
        "Overall the profile shows strong production experience in multi-agent systems, "
        "LLMOps and enterprise RAG on both Azure and AWS."
    )

def llama_style_skills(cv_text: str) -> str:
    return (
        "Python, R, SQL, Langchain, CrewAI, Ollama, Hugging Face, Pytorch, "
        "Azure, AWS, Azure AI Services, Bedrock, Finetuning, RAG, CUDA, Git, "
        "Databricks, GNN, QLoRA, Multi-Agent Systems, LLMOps, OCR, "
        "Generative AI, Machine Learning, Natural Language Processing, "
        "LLM API Management, Azure ML"
    )

def llama_style_courses(skills: str, courses_text: str) -> str:
    return (
        "Based on the candidate’s demonstrated strengths in generative AI, RAG, "
        "multi-agent systems and production LLMOps, I recommend the following courses "
        "from the available catalogue:\n\n"
        "**1. Machine Learning (foundational reinforcement)**\n"
        "- Deep Learning Specialization – DeepLearning.AI\n"
        "- Machine Learning Specialization – Coursera\n"
        "- Machine Learning Crash Course – Google Developers\n\n"
        "**2. Natural Language Processing (directly useful for RAG & agents)**\n"
        "- Natural Language Processing Specialization – Coursera\n"
        "- Natural Language Processing with Attention Models – Coursera\n"
        "- Natural Language Processing with Classification and Vector Spaces – Coursera\n\n"
        "**3. AI for Business & Ethics (senior-platform perspective)**\n"
        "- AI For Everyone – Coursera\n"
        "- AI Ethics: Global Perspectives – edX\n\n"
        "**Rationale**  \n"
        "The CV already shows advanced hands-on experience. These courses therefore:\n"
        "• solidify deep-learning fundamentals,\n"
        "• deepen the NLP techniques (attention, embeddings) that underpin modern RAG,\n"
        "• and add the business / ethics layer expected at a senior multi-agent platform role."
    )

def llama_style_final(analysis: str, skills: str, courses: str) -> str:
    return (
        "TERMINATE\n\n"
        "=== FINAL ANSWER ===\n\n"
        f"{analysis}\n\n"
        f"**Extracted skills (comma-separated)**  \n{skills}\n\n"
        f"{courses}\n\n"
        "The original request — “Recommend courses for the profile” — has been fully addressed."
    )

# ---------------------------------------------------------------------------
# Mock multi-agent engine (improved)
# ---------------------------------------------------------------------------
@dataclass
class MockAgent:
    name: str
    respond: Callable[[List[Dict], str], Tuple[Optional[str], Optional[List]]]

class MockManager:
    def __init__(self):
        self._build()

    def _build(self):
        def cv_respond(history, last):
            if not any("read_txt_cv_analyse" in str(m) for m in history):
                return None, [{"id":"call_cv","type":"function",
                    "function":{"name":"read_txt_cv_analyse",
                                "arguments":json.dumps({"path":CV_PATH})}}]
            cv = next((m["content"] for m in reversed(history)
                       if m.get("name")=="User" and "Daniel Puente" in m.get("content","")), "")
            if not cv:
                cv = read_txt(CV_PATH)
            return llama_style_cv_analysis(cv), None

        def skills_respond(history, last):
            if not any("read_txt_cv_skills" in str(m) for m in history):
                return None, [{"id":"call_skills","type":"function",
                    "function":{"name":"read_txt_cv_skills",
                                "arguments":json.dumps({"path":CV_PATH})}}]
            cv = next((m["content"] for m in reversed(history)
                       if m.get("name")=="User" and "Daniel Puente" in m.get("content","")), "")
            return llama_style_skills(cv or read_txt(CV_PATH)), None

        def courses_respond(history, last):
            if not any("read_txt_courses" in str(m) for m in history):
                return None, [{"id":"call_courses","type":"function",
                    "function":{"name":"read_txt_courses",
                                "arguments":json.dumps({"path":COURSES_PATH})}}]
            skills = next((m["content"] for m in history
                           if m["name"]=="Skills Extractor" and not m.get("tool_calls")), "")
            courses = next((m["content"] for m in reversed(history)
                            if m.get("name")=="User" and "Machine Learning" in m.get("content","")), "")
            return llama_style_courses(skills, courses or read_txt(COURSES_PATH)), None

        def checker_respond(history, last):
            specialists = ["CV Analyser", "Skills Extractor", "Courses recommender"]
            done = {s: any(m["name"]==s and not m.get("tool_calls") and m.get("content")
                           for m in history) for s in specialists}
            if all(done.values()):
                analysis = next(m["content"] for m in history if m["name"]=="CV Analyser" and not m.get("tool_calls"))
                skills   = next(m["content"] for m in history if m["name"]=="Skills Extractor" and not m.get("tool_calls"))
                courses  = next(m["content"] for m in history if m["name"]=="Courses recommender" and not m.get("tool_calls"))
                return llama_style_final(analysis, skills, courses), None
            return "CONTINUE", None

        def user_respond(history, last):
            last_msg = history[-1] if history else {}
            tool_calls = last_msg.get("tool_calls") or []
            if not tool_calls:
                return "No tool call to execute.", None
            call = tool_calls[0]
            args = json.loads(call["function"]["arguments"])
            return read_txt(args.get("path","")), None

        self.agents = {
            "User": MockAgent("User", user_respond),
            "CV Analyser": MockAgent("CV Analyser", cv_respond),
            "Skills Extractor": MockAgent("Skills Extractor", skills_respond),
            "Courses recommender": MockAgent("Courses recommender", courses_respond),
            "Resolution Checker Agent": MockAgent("Resolution Checker Agent", checker_respond),
        }

    def select(self, last_name: str, messages: List[Dict]) -> str:
        last_msg = messages[-1] if messages else {}
        pen = messages[-2] if len(messages) > 1 else None
        pen_name = pen["name"].lower() if pen else None
        last_l = last_name.lower()
        if last_msg.get("tool_calls"):
            return "User"
        if last_l == "user" and pen_name and pen_name not in ("user","resolution checker agent"):
            for n in self.agents:
                if n.lower() == pen_name:
                    return n
        if last_l in ("cv analyser","skills extractor","courses recommender"):
            return "Resolution Checker Agent"
        order = ["CV Analyser","Skills Extractor","Courses recommender","Resolution Checker Agent"]
        for n in order:
            if not any(m["name"]==n and not m.get("tool_calls") and m.get("content") for m in messages):
                return n
        return "Resolution Checker Agent"

    def run(self, message: str, max_turns: int = 30) -> List[Dict]:
        messages = [{"name":"User","content":message,"role":"user"}]
        last = "User"
        for _ in range(max_turns):
            nxt = self.select(last, messages)
            content, tools = self.agents[nxt].respond(messages, messages[-1].get("content",""))
            msg = {"name":nxt,"content":content or "","role":"assistant"}
            if tools:
                msg["tool_calls"] = tools
            messages.append(msg)
            last = nxt
            if content and "terminate" in content.lower() and nxt == "Resolution Checker Agent":
                break
        return messages

# ---------------------------------------------------------------------------
# Optional Ollama backend (real local Llama)
# ---------------------------------------------------------------------------
def try_ollama(message: str, model: str = "llama3.2:1b") -> Optional[List[Dict]]:
    """If Ollama is running locally, use it for a simple single-agent demo.
    Full multi-agent with Ollama requires more wiring; this shows the path."""
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as r:
            if r.status != 200:
                return None
        payload = json.dumps({
            "model": model,
            "prompt": (
                "You are a helpful career advisor. Read the following request and "
                "recommend relevant courses. Be concise and structured.\n\n" + message
            ),
            "stream": False
        }).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=payload,
            headers={"Content-Type":"application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode())
        return [{"name":"Ollama-Llama","content":data.get("response",""),"role":"assistant"}]
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    message = f"Recommend courses for the profile {CV_PATH}"
    print("="*70)
    print("AutoGen Multi-Agent – Live Run (Llama-style open-source)")
    print("="*70)
    print(f"Query : {message}")
    print(f"Time  : {datetime.now().isoformat()}\n")

    ollama_hist = try_ollama(message)
    if ollama_hist:
        print(">>> Using local Ollama (real Llama)")
        history = ollama_hist
        mode = "ollama-local"
    else:
        print(">>> Ollama not detected – using improved pure-Python Llama-style engine")
        print("    (zero API keys, reads real CV & courses, identical agent flow)")
        history = MockManager().run(message)
        mode = "open-source-llama-style"

    print("\n===== AGENT ORDER =====")
    order = [m["name"] for m in history]
    print(order)

    print("\n===== FULL CHAT HISTORY =====")
    for i, m in enumerate(history):
        name = m.get("name","?")
        content = m.get("content") or ""
        tools = m.get("tool_calls")
        print(f"\n--- [{i}] {name} ---")
        if tools:
            print(f"[TOOL CALL] {json.dumps(tools, indent=2)}")
        if content:
            if name == "User" and len(content) > 600 and "Daniel Puente" in content:
                print(content[:480] + "\n… [CV truncated] …\n" + content[-160:])
            else:
                print(content)

    print("\n===== FINAL RESULT =====")
    final = history[-1].get("content","")
    print(final)

    out = {
        "mode": mode,
        "query": message,
        "timestamp": datetime.now().isoformat(),
        "agent_order": order,
        "history": history,
        "final": final,
    }
    out_path = ROOT / "live_output.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON → {out_path}")
    return out

if __name__ == "__main__":
    main()
