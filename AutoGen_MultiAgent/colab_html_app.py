"""Colab-friendly HTML UI for the AutoGen_MultiAgent profile workflow.

Uses an open-weight Hugging Face chat model through Transformers. No API key is
required. Course titles shown as recommendations are selected from the supplied
catalogue by validating model-returned catalogue IDs.
"""
from __future__ import annotations

import gc
import html
import json
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_CHOICES = [
    ("Qwen2.5 0.5B Instruct — lightest (recommended)", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("Qwen2.5 1.5B Instruct — better quality, more memory", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("TinyLlama 1.1B Chat — alternative", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
]
DEFAULT_MODEL = MODEL_CHOICES[0][1]

SAMPLE_COURSES = """Machine Learning Specialization — DeepLearning.AI / Coursera
Deep Learning Specialization — DeepLearning.AI
Machine Learning Crash Course — Google for Developers
Natural Language Processing Specialization — Coursera
Natural Language Processing with Attention Models — Coursera
AI For Everyone — DeepLearning.AI / Coursera
Practical Deep Learning for Coders — fast.ai
Hugging Face NLP Course — Hugging Face
"""

_MODEL_LOCK = threading.RLock()
_RUN_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, Any] = {"id": None, "tokenizer": None, "model": None, "torch": None, "device": None}


def _read_uploaded_text(filepath: str | None) -> str:
    """Read supported text files; PDF/DOCX support is loaded only when needed."""
    if not filepath:
        return ""
    path = Path(str(filepath))
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf to read PDF files: pip install pypdf") from exc
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("Install python-docx to read DOCX files: pip install python-docx") from exc
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Unsupported file type. Upload .txt, .md, .csv, .pdf, or .docx.")


def _catalog_entries(catalogue: str, limit: int = 60) -> list[str]:
    """Treat each non-heading line in the catalogue as a selectable course item."""
    entries: list[str] = []
    seen: set[str] = set()
    for raw in (catalogue or "").splitlines():
        line = raw.strip()
        line = re.sub(r"^\s*#+\s*", "", line)
        line = re.sub(r"^\s*(?:[-*•]\s*|\d+[.)]\s*)", "", line).strip()
        heading = line.casefold().rstrip(":").strip()
        if (not line or len(line) < 3 or (line.endswith(":") and len(line) < 80)
                or heading in {"course", "courses", "course catalog", "course catalogue", "catalog", "catalogue", "available courses"}):
            continue
        line = line[:400]
        key = line.casefold()
        if key not in seen:
            entries.append(line)
            seen.add(key)
        if len(entries) >= limit:
            break
    return entries


def _parse_course_recommendations(raw: str, entries: list[str]) -> list[dict[str, str]]:
    """Validate model-selected IDs and use canonical course names from the input."""
    if not entries or not raw:
        return []
    candidates: list[dict[str, Any]] = []
    try:
        match = re.search(r"\{[\s\S]*\}", raw)
        payload = json.loads(match.group(0)) if match else {}
        values = payload.get("recommendations", []) if isinstance(payload, dict) else []
        if isinstance(values, list):
            candidates = [x for x in values if isinstance(x, dict)]
    except (ValueError, TypeError, json.JSONDecodeError):
        candidates = []

    if not candidates:
        candidates = [{"id": code, "reason": "Selected by the model; rationale unavailable."}
                      for code in re.findall(r"\bC\d{1,3}\b", raw, flags=re.IGNORECASE)]

    validated: list[dict[str, str]] = []
    seen: set[int] = set()
    for item in candidates:
        code = str(item.get("id", "")).upper().strip()
        match_id = re.fullmatch(r"C(\d{1,3})", code)
        if not match_id:
            continue
        index = int(match_id.group(1))
        if index < 1 or index > len(entries) or index in seen:
            continue
        reason = str(item.get("reason", "")).strip()
        if len(reason) > 500:
            reason = reason[:500] + "…"
        validated.append({"id": f"C{index:02d}", "title": entries[index - 1],
                          "reason": reason or "The model did not provide a rationale."})
        seen.add(index)
    return validated[:8]


def render_report_html(candidate_name: str, model_id: str, device: str,
                       sections: list[tuple[str, str]]) -> str:
    """Create a self-contained, escaped HTML report suitable for preview/download."""
    safe_name = html.escape(candidate_name or "Candidate")
    safe_model = html.escape(model_id)
    safe_device = html.escape(device)
    cards = []
    for title, body in sections:
        cards.append(
            '<section class="card"><h2>' + html.escape(title) + '</h2><div class="body">'
            + html.escape(body or "No output was produced.").replace("\n", "<br>")
            + "</div></section>"
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-Agent Profile Report</title>
<style>
:root{color-scheme:light;--ink:#172033;--muted:#5d6b82;--line:#dbe2ec;--panel:#fff;--bg:#f4f7fb;--accent:#2856a6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1060px;margin:32px auto;padding:0 18px}.hero{background:linear-gradient(135deg,#12213e,#2e5594);color:white;border-radius:18px;padding:28px 30px;margin-bottom:20px}
h1{font-size:clamp(1.6rem,4vw,2.3rem);line-height:1.2;margin:0 0 10px}.meta{color:#e0e9ff;font-size:.92rem}
.note{border-left:4px solid #d39a27;background:#fff8e6;padding:12px 15px;margin:16px 0 22px;border-radius:8px;color:#594319}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:16px}.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 4px 16px #1720330a}
h2{margin:0 0 10px;font-size:1.15rem;color:var(--accent)}.body{overflow-wrap:anywhere;white-space:normal}
footer{color:var(--muted);font-size:.86rem;margin:20px 2px}@media print{body{background:white}main{margin:0 auto}.hero{print-color-adjust:exact}}
</style></head><body><main>
<header class="hero"><h1>Multi-Agent Profile Analysis</h1><div class="meta">Candidate: """ + safe_name + """</div>
<div class="meta">Model: """ + safe_model + """ · Runtime: """ + safe_device + """ · Generated: """ + timestamp + """</div></header>
<div class="note"><strong>Review required.</strong> This report is AI-generated and may contain errors. Confirm profile facts and recommendations before relying on them. Course titles below are taken from the supplied catalogue only when a valid catalogue ID was returned.</div>
<div class="grid">""" + "".join(cards) + """</div>
<footer>Generated locally in the Colab runtime using an open-weight model. This report does not independently verify employment, qualifications, course availability, or external facts.</footer>
</main></body></html>"""


def _load_model_locked(model_id: str):
    """Load one model at a time to avoid filling limited Colab RAM/VRAM."""
    if _MODEL_CACHE["id"] == model_id and _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Runtime packages are missing. Run the notebook setup cell to install "
            "transformers, accelerate, and torch."
        ) from exc

    old_torch = _MODEL_CACHE.get("torch")
    _MODEL_CACHE.update({"id": None, "tokenizer": None, "model": None, "torch": None, "device": None})
    gc.collect()
    if old_torch is not None and old_torch.cuda.is_available():
        old_torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            trust_remote_code=False,
        )
        model.to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {model_id}. Check the internet connection and available "
            f"Colab RAM/VRAM, or select the smaller Qwen2.5 0.5B model. Detail: {exc}"
        ) from exc
    _MODEL_CACHE.update({"id": model_id, "tokenizer": tokenizer, "model": model,
                         "torch": torch, "device": device})
    return _MODEL_CACHE


def _generate(model_id: str, system_prompt: str, user_prompt: str,
              max_new_tokens: int = 220) -> tuple[str, str]:
    with _MODEL_LOCK:
        cached = _load_model_locked(model_id)
        tokenizer, model, torch = cached["tokenizer"], cached["model"], cached["torch"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                truncation=True, max_length=6000, return_tensors="pt",
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=6000)["input_ids"]
        input_ids = input_ids.to(model.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
        return text, cached["device"]


def run_analysis(candidate_name: str, target_role: str, cv_text: str,
                 catalogue_text: str, model_id: str = DEFAULT_MODEL) -> tuple[str, str, str]:
    """Run role-separated LLM steps and return HTML, report path, and status."""
    cv_text = (cv_text or "").strip()
    if not cv_text:
        raise ValueError("Please paste CV/profile text or upload a CV file.")
    if model_id not in {item[1] for item in MODEL_CHOICES}:
        raise ValueError("Please choose one of the supported open-weight models.")
    candidate_name = (candidate_name or "").strip() or "Candidate"
    target_role = (target_role or "").strip() or "Not specified"
    cv_text = cv_text[:18000]
    catalogue_text = (catalogue_text or "").strip()[:14000]
    entries = _catalog_entries(catalogue_text)

    analyst_system = (
        "You are the CV Analyst agent. Use only the supplied CV. Never invent employers, "
        "dates, degrees, metrics, or achievements. Separate explicit evidence from inference. "
        "Write a concise report with profile summary, experience, education, strengths, and "
        "missing information. Say 'Not stated in the provided CV' when needed."
    )
    skills_system = (
        "You are the Skills Extractor agent. Extract only skills supported by the provided CV. "
        "Group technical skills, AI/data skills, cloud/platform skills, and professional skills. "
        "For important skills include a short evidence phrase. Do not add guessed skills."
    )
    courses_system = (
        "You are the Course Recommender agent. Select only relevant IDs from the exact catalogue "
        "provided. Do not create course names or use outside knowledge. Return ONLY valid JSON "
        'of the form {"recommendations":[{"id":"C01","reason":"short evidence-based reason"}]}. '
        "Select at most 5 items. If none fit, return {\"recommendations\":[]}."
    )
    checker_system = (
        "You are the Resolution Checker agent. Audit the draft sections for unsupported CV claims, "
        "overconfident inferences, and recommendations needing human verification. Do not rewrite "
        "the report, add facts, or suggest any new course. Return short actionable checks only."
    )

    with _RUN_LOCK:
        analyst, device = _generate(
            model_id, analyst_system,
            f"Candidate label: {candidate_name}\nTarget role: {target_role}\nCV text:\n{cv_text}",
            max_new_tokens=300,
        )
        skills, _ = _generate(
            model_id, skills_system,
            f"Target role: {target_role}\nCV text:\n{cv_text}",
            max_new_tokens=240,
        )

        if entries:
            catalogue_with_ids = "\n".join(f"[C{i:02d}] {entry}" for i, entry in enumerate(entries, 1))
            course_draft, _ = _generate(
                model_id, courses_system,
                f"Target role: {target_role}\nCV analysis:\n{analyst[:5000]}\n"
                f"Extracted skills:\n{skills[:4000]}\nCatalogue entries (the IDs are authoritative):\n"
                f"{catalogue_with_ids}",
                max_new_tokens=240,
            )
            selected = _parse_course_recommendations(course_draft, entries)
            if selected:
                course_result = "\n".join(
                    f"{i}. {item['title']} [{item['id']}]\n   Rationale: {item['reason']}"
                    for i, item in enumerate(selected, 1)
                )
                course_result += "\n\nValidation: each displayed title maps to a supplied catalogue ID."
            else:
                course_result = (
                    "No catalogue-backed recommendations were displayed. The model did not return "
                    "valid catalogue IDs, so its draft was suppressed rather than showing unverified "
                    "course names. Try simplifying the catalogue to one course per line."
                )
        else:
            course_result = (
                "No usable course catalogue entries were supplied. Add a catalogue with one course "
                "per line; this app does not present invented course titles as catalogue-backed."
            )

        checker, _ = _generate(
            model_id, checker_system,
            f"Target role: {target_role}\nCV analysis:\n{analyst[:4500]}\n"
            f"Skills output:\n{skills[:3500]}\nCourse recommendations:\n{course_result[:2500]}\n"
            "Identify what requires manual checking. Do not introduce new facts or course names.",
            max_new_tokens=160,
        )

    sections = [
        ("1 · CV Analyst", analyst),
        ("2 · Skills Extractor", skills),
        ("3 · Courses Recommender", course_result),
        ("4 · Resolution Checker", checker),
        ("Verification notes", (
            "Open-weight model inference ran in the Colab runtime. The CV was treated as input text, "
            "not independently verified. Course titles were displayed only when the model selected "
            "valid IDs mapped to entries in the supplied catalogue. Review all outputs manually."
        )),
    ]
    report_html = render_report_html(candidate_name, model_id, device, sections)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".html",
                                     prefix="autogen_profile_report_", delete=False) as handle:
        handle.write(report_html)
        report_path = handle.name
    return report_html, report_path, f"Completed using {model_id} on {device.upper()}. Review the report before relying on it."


def _preview_fragment(document: str) -> str:
    """Convert the standalone document to a fragment suitable for Gradio's HTML component."""
    style = re.search(r"<style[^>]*>([\s\S]*?)</style>", document, flags=re.IGNORECASE)
    body = re.search(r"<body[^>]*>([\s\S]*?)</body>", document, flags=re.IGNORECASE)
    if not body:
        return document
    style_block = f"<style>{style.group(1)}</style>" if style else ""
    return style_block + body.group(1)


def _process_form(candidate_name, target_role, model_id, cv_upload, cv_text,
                  catalogue_upload, catalogue_text):
    try:
        cv = _read_uploaded_text(cv_upload) if cv_upload else (cv_text or "")
        catalogue = _read_uploaded_text(catalogue_upload) if catalogue_upload else (catalogue_text or "")
        report, file_path, status = run_analysis(candidate_name, target_role, cv, catalogue, model_id)
        return _preview_fragment(report), file_path, status
    except Exception as exc:
        message = html.escape(str(exc))
        error_html = (
            '<div style="font-family:system-ui;padding:18px;border:1px solid #e7a5a5;'
            'border-radius:12px;background:#fff4f4;color:#7b1f1f"><h3>Unable to complete analysis</h3>'
            f'<p>{message}</p></div>'
        )
        return error_html, None, f"Analysis not completed: {exc}"


def build_demo():
    """Build the browser-based HTML form. Gradio is required only when launching the UI."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install Gradio with: pip install 'gradio>=5.23,<6'") from exc

    css = """
    .app-title{font-size:1.8rem!important}
    .privacy-note{border-left:4px solid #c18b20;padding:10px 14px;background:#fff8e8;border-radius:8px}
    footer{color:#667085}
    """
    with gr.Blocks(title="Open-Weight Multi-Agent Profile Analyzer", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown("# Open-Weight Multi-Agent Profile Analyzer\n"
                    "Colab-ready HTML app · no paid API key · downloadable report")
        gr.Markdown(
            "**Privacy:** the Colab/Gradio share link is public while running. Use a redacted/demo CV; "
            "do not submit sensitive personal information. Processing runs in your Colab runtime."
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                candidate = gr.Textbox(label="Candidate label", placeholder="e.g. Candidate A")
                target = gr.Textbox(label="Target role / learning goal", placeholder="e.g. Senior AI Engineer")
                model = gr.Dropdown(
                    choices=MODEL_CHOICES, value=DEFAULT_MODEL, label="Open-weight LLM",
                    info="The first run downloads the chosen model. GPU runtime is recommended.",
                )
                gr.Markdown("### CV / profile")
                cv_upload = gr.File(label="Upload CV (optional)", file_types=[".txt", ".md", ".csv", ".pdf", ".docx"], type="filepath")
                cv_text = gr.Textbox(label="Or paste CV/profile text", lines=12,
                                      placeholder="Paste a redacted CV or professional profile here.")
                gr.Markdown("### Course catalogue")
                course_upload = gr.File(label="Upload catalogue (optional)", file_types=[".txt", ".md", ".csv", ".pdf", ".docx"], type="filepath")
                courses = gr.Textbox(label="Or paste catalogue — one course per line",
                                     value=SAMPLE_COURSES, lines=8)
                submit = gr.Button("Analyze profile and generate HTML report", variant="primary")
            with gr.Column(scale=1, min_width=320):
                status = gr.Markdown("Ready. Add a CV/profile and run the analysis.")
                report = gr.HTML(label="HTML report")
                download = gr.File(label="Download standalone HTML report")
        submit.click(
            fn=_process_form,
            inputs=[candidate, target, model, cv_upload, cv_text, course_upload, courses],
            outputs=[report, download, status],
        )
        gr.Markdown(
            "Models: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), "
            "[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), "
            "[TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0). "
            "Model availability and speed depend on Colab resources."
        )
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.queue(max_size=4)
    app.launch(share=True, show_error=True)
