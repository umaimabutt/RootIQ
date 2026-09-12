import os
import io
import json
import base64
from datetime import datetime
from typing import Any, Dict, List
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from docx import Document

APP_TITLE = "RootIQ — AI Incident Investigation & Root Cause Analysis"

# Streamlit Community Cloud reads secrets configured in the app settings.
# For local development, the same values can live in .streamlit/secrets.toml.
# Environment variables remain supported as a fallback.
def get_secret(name: str, default: str | None = None) -> str | None:
    try:
        value = st.secrets.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        # st.secrets may not exist in a plain local Python environment.
        pass
    value = os.getenv(name)
    return value.strip() if value else default


DEFAULT_MODEL = get_secret("ROOTIQ_MODEL", "gpt-5.6-luna")

st.set_page_config(
    page_title="RootIQ",
    page_icon="🔎",
    layout="wide",
)

# -----------------------------
# Session state
# -----------------------------
DEFAULT_STATE = {
    "evidence": [],
    "analysis": None,
    "history": [],
    "investigation_title": "",
    "last_follow_up": None,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> OpenAI:
    """Create one reusable OpenAI client per API key for the Streamlit session/app."""
    return OpenAI(api_key=api_key)


def get_openai_client() -> OpenAI:
    # Streamlit Cloud: st.secrets -> local environment variable.
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add it to Streamlit Cloud Secrets "
            "or to .streamlit/secrets.toml for local development."
        )
    return get_client(api_key)


def clean_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def text_from_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(f"[PDF page {i + 1}]\n{page.extract_text() or ''}")
        except Exception as exc:
            pages.append(f"[PDF page {i + 1} extraction failed: {exc}]")
    return "\n\n".join(pages)


def text_from_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def text_from_csv(data: bytes) -> str:
    df = pd.read_csv(io.BytesIO(data))
    return df.to_csv(index=False)


def text_from_excel(data: bytes) -> str:
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
    parts = []
    for name, df in sheets.items():
        parts.append(f"[Excel sheet: {name}]\n{df.to_csv(index=False)}")
    return "\n\n".join(parts)


def image_to_data_url(data: bytes, mime: str) -> str:
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def parse_uploaded_file(uploaded_file) -> Dict[str, Any]:
    name = uploaded_file.name
    mime = uploaded_file.type or ""
    data = uploaded_file.getvalue()
    ext = Path(name).suffix.lower()

    result = {
        "name": name,
        "mime": mime,
        "size": len(data),
        "type": "text",
        "text": "",
        "image_data_url": None,
    }

    if ext == ".pdf":
        result["text"] = text_from_pdf(data)
    elif ext in {".xlsx", ".xls"}:
        result["text"] = text_from_excel(data)
    elif ext == ".csv":
        result["text"] = text_from_csv(data)
    elif ext in {".docx"}:
        result["text"] = text_from_docx(data)
    elif ext in {".txt", ".log", ".md", ".json", ".xml", ".html", ".eml"}:
        result["text"] = data.decode("utf-8", errors="replace")
    elif ext in {".png", ".jpg", ".jpeg", ".webp"} or mime.startswith("image/"):
        result["type"] = "image"
        result["image_data_url"] = image_to_data_url(data, mime or "image/png")
    else:
        # Try UTF-8 for unknown text-like files.
        try:
            result["text"] = data.decode("utf-8", errors="replace")
        except Exception:
            result["text"] = f"Unsupported file type: {ext}"

    return result


def evidence_context() -> str:
    chunks = []
    for item in st.session_state.evidence:
        if item["type"] == "image":
            chunks.append(f"[IMAGE EVIDENCE: {item['name']}]")
        else:
            text = item.get("text", "")
            chunks.append(f"[EVIDENCE: {item['name']}]\n{text[:30000]}")
    return "\n\n---\n\n".join(chunks)


def image_inputs() -> List[Dict[str, Any]]:
    items = []
    for item in st.session_state.evidence:
        if item["type"] == "image" and item.get("image_data_url"):
            items.append(
                {
                    "type": "input_image",
                    "image_url": item["image_data_url"],
                    "detail": "high",
                }
            )
    return items


SYSTEM_PROMPT = """
You are RootIQ, an evidence-first incident investigation and root-cause analysis engine.

Your job is NOT to summarize evidence. You must reconstruct events, reason about causal relationships,
estimate impact when supported, identify root-cause hypotheses, and recommend corrective/preventive actions.

Hard rules:
1. Never invent missing facts.
2. Correlation is not proof of causation.
3. Every major conclusion must identify supporting evidence by source filename.
4. Clearly distinguish: Confirmed cause, Strongly supported cause, Possible cause,
   Correlated factor, and Insufficient evidence.
5. If evidence is missing, explicitly ask for the most useful additional evidence.
6. When new evidence is provided, REASSESS the entire hypothesis rather than merely appending it.
7. If new evidence contradicts an earlier hypothesis, say so and reduce/change its confidence.
8. Estimates and what-if outcomes must be labeled as estimates/scenarios.
9. Prefer conservative conclusions over unsupported certainty.
10. The principle is: No Evidence -> No Strong Conclusion.
"""


ANALYSIS_SCHEMA_INSTRUCTIONS = """
Return ONLY valid JSON with this top-level shape:

{
  "incident_summary": "...",
  "severity": "LOW|MEDIUM|HIGH|CRITICAL|UNKNOWN",
  "severity_reason": "...",
  "key_metrics": [
    {"metric": "...", "value": "...", "source": "..."}
  ],
  "timeline": [
    {"time": "...", "event": "...", "source": "...", "confidence": "High|Medium|Low"}
  ],
  "causal_chain": [
    {"from": "...", "to": "...", "relationship": "causal|possible_causal|correlated|unknown", "evidence": ["filename"], "confidence": "High|Medium|Low"}
  ],
  "root_cause_hypotheses": [
    {
      "hypothesis": "...",
      "status": "Confirmed cause|Strongly supported cause|Possible cause|Correlated factor|Insufficient evidence",
      "confidence": "High|Medium|Low",
      "supporting_evidence": ["filename or exact evidence description"],
      "contradicting_evidence": ["..."],
      "reasoning": "..."
    }
  ],
  "impact": {
    "financial": [{"metric": "...", "value": "...", "evidence": "..."}],
    "users": [{"metric": "...", "value": "...", "evidence": "..."}],
    "operational": [{"metric": "...", "value": "...", "evidence": "..."}]
  },
  "corrective_actions": ["..."],
  "preventive_actions": ["..."],
  "what_if_scenarios": [
    {"question": "...", "scenario": "...", "result": "...", "confidence": "High|Medium|Low", "basis": "..."}
  ],
  "limitations": ["..."],
  "follow_up_questions": [
    {
      "question": "Please provide ...",
      "why_needed": "...",
      "expected_value": "What this evidence could confirm or rule out"
    }
  ],
  "hypothesis_change_note": "Explain what changed compared with any prior analysis, or say 'Initial analysis.'"
}
"""


def build_analysis_prompt(previous_analysis: Dict[str, Any] | None = None) -> str:
    prior = ""
    if previous_analysis:
        prior = (
            "\n\nPREVIOUS ANALYSIS — treat this only as a prior hypothesis, not as fact. "
            "You MUST test it against all current evidence:\n"
            + json.dumps(previous_analysis, ensure_ascii=False, indent=2)[:50000]
        )

    return f"""
{SYSTEM_PROMPT}

{ANALYSIS_SCHEMA_INSTRUCTIONS}

Analyze the current investigation evidence below.
{prior}

CURRENT EVIDENCE:
{evidence_context()}
"""


def call_text_model(prompt: str, include_images: bool = True) -> str:
    client = get_openai_client()
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    if include_images:
        content.extend(image_inputs())

    response = client.responses.create(
        model=DEFAULT_MODEL,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    )
    return response.output_text


def run_investigation() -> Dict[str, Any]:
    if not st.session_state.evidence:
        raise ValueError("Upload at least one evidence file before running RootIQ.")
    raw = call_text_model(build_analysis_prompt(st.session_state.analysis))
    return clean_json(raw)


def chat_with_investigation(question: str) -> str:
    client = get_openai_client()
    prompt = f"""
You are the RootIQ investigation chatbot.

Answer the user's question using ONLY the evidence and latest analysis below.
If the answer is not supported, say that clearly and identify what evidence is missing.
Cite source filenames naturally in your answer.

LATEST ANALYSIS:
{json.dumps(st.session_state.analysis or {}, ensure_ascii=False, indent=2)[:50000]}

RAW EVIDENCE:
{evidence_context()[:60000]}

USER QUESTION:
{question}
"""
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    content.extend(image_inputs())

    response = client.responses.create(
        model=DEFAULT_MODEL,
        input=[{"role": "user", "content": content}],
    )
    return response.output_text


def run_what_if(question: str) -> str:
    client = get_openai_client()
    prompt = f"""
Use RootIQ's evidence-first principles.

Investigation analysis:
{json.dumps(st.session_state.analysis or {}, ensure_ascii=False, indent=2)[:50000]}

Evidence:
{evidence_context()[:60000]}

What-if question:
{question}

Explain:
1. What the evidence supports.
2. The counterfactual scenario.
3. Expected impact, if estimable.
4. Assumptions and uncertainty.
Never present a scenario estimate as a historical fact.
"""
    response = client.responses.create(
        model=DEFAULT_MODEL,
        input=prompt,
    )
    return response.output_text


def report_markdown() -> str:
    a = st.session_state.analysis or {}
    lines = [
        f"# RootIQ Investigation Report",
        "",
        f"**Investigation:** {st.session_state.investigation_title or 'Untitled investigation'}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Incident Summary",
        a.get("incident_summary", "Not available"),
        "",
        f"## Severity: {a.get('severity', 'UNKNOWN')}",
        a.get("severity_reason", ""),
        "",
        "## Root Cause Hypotheses",
    ]
    for h in a.get("root_cause_hypotheses", []):
        lines.extend(
            [
                f"### {h.get('status', 'Hypothesis')} — {h.get('confidence', 'Unknown')} confidence",
                h.get("hypothesis", ""),
                f"**Reasoning:** {h.get('reasoning', '')}",
                f"**Supporting evidence:** {', '.join(h.get('supporting_evidence', [])) or 'None identified'}",
                f"**Contradicting evidence:** {', '.join(h.get('contradicting_evidence', [])) or 'None identified'}",
                "",
            ]
        )

    lines.append("## Timeline")
    for e in a.get("timeline", []):
        lines.append(
            f"- **{e.get('time', '?')}** — {e.get('event', '')} "
            f"({e.get('source', 'source not identified')}; {e.get('confidence', 'Unknown')} confidence)"
        )

    lines.extend(["", "## Impact"])
    for category, values in (a.get("impact") or {}).items():
        lines.append(f"### {category.title()}")
        for x in values or []:
            lines.append(f"- {x.get('metric', '')}: {x.get('value', '')} — {x.get('evidence', '')}")

    lines.extend(["", "## Corrective Actions"])
    lines.extend([f"- {x}" for x in a.get("corrective_actions", [])])
    lines.extend(["", "## Preventive Actions"])
    lines.extend([f"- {x}" for x in a.get("preventive_actions", [])])

    lines.extend(["", "## Follow-up Evidence Requested"])
    for q in a.get("follow_up_questions", []):
        lines.append(f"- {q.get('question', '')} — {q.get('why_needed', '')}")

    lines.extend(["", "## Limitations"])
    lines.extend([f"- {x}" for x in a.get("limitations", [])])

    return "\n".join(lines)


# -----------------------------
# UI
# -----------------------------
st.title("🔎 RootIQ")
st.caption(APP_TITLE)
st.caption("Evidence → Timeline → Causal Analysis → Impact → Root Cause → What-If → Prevention")

with st.sidebar:
    st.header("Investigation")
    st.session_state.investigation_title = st.text_input(
        "Investigation title",
        value=st.session_state.investigation_title,
        placeholder="e.g. Online payment failure",
    )

    api_configured = bool(get_secret("OPENAI_API_KEY"))
    if api_configured:
        st.success("OpenAI API configured", icon="✅")
    else:
        st.warning(
            "OpenAI API key not configured. Add OPENAI_API_KEY in Streamlit Secrets.",
            icon="⚠️",
        )

    st.divider()
    st.write(f"**Model:** `{DEFAULT_MODEL}`")
    st.write(f"**Evidence files:** {len(st.session_state.evidence)}")
    st.caption("Deployment: GitHub → Streamlit Community Cloud")

    if st.button("Reset investigation", use_container_width=True):
        for key, value in DEFAULT_STATE.items():
            st.session_state[key] = value
        st.rerun()


tabs = st.tabs(
    [
        "📥 Evidence",
        "🧠 Investigation",
        "❓ Follow-up evidence",
        "💬 Chat",
        "🔮 What-if",
        "📄 Report",
    ]
)

# Evidence
with tabs[0]:
    st.subheader("Upload investigation evidence")
    st.write(
        "RootIQ accepts CSV/XLSX, PDF, text/log/email files, DOCX, and screenshots/images."
    )

    uploads = st.file_uploader(
        "Add evidence",
        type=[
            "csv", "xlsx", "xls", "pdf", "txt", "log", "md", "json",
            "xml", "html", "eml", "docx", "png", "jpg", "jpeg", "webp"
        ],
        accept_multiple_files=True,
    )

    if st.button("➕ Add uploaded evidence", disabled=not uploads):
        existing_names = {x["name"] for x in st.session_state.evidence}
        added = 0
        for file in uploads:
            if file.name in existing_names:
                continue
            st.session_state.evidence.append(parse_uploaded_file(file))
            added += 1
        st.success(f"Added {added} evidence file(s).")
        st.rerun()

    if st.session_state.evidence:
        st.subheader("Current evidence")
        for i, item in enumerate(st.session_state.evidence):
            c1, c2, c3 = st.columns([5, 2, 1])
            c1.write(f"**{item['name']}**")
            c2.write(f"{item['size'] / 1024:.1f} KB")
            if c3.button("Remove", key=f"remove_{i}"):
                st.session_state.evidence.pop(i)
                st.rerun()

        st.info(
            "You can add more evidence at any time. When new evidence is added, "
            "RootIQ re-runs the analysis and explicitly compares the new evidence "
            "with its previous hypotheses."
        )

# Investigation
with tabs[1]:
    st.subheader("RootIQ investigation")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(
            "Run the investigation after uploading evidence. RootIQ will reconstruct "
            "the timeline, assess impact/severity, generate hypotheses, and identify "
            "missing evidence."
        )
    with col2:
        run = st.button(
            "🚀 Run / Update RootIQ",
            type="primary",
            disabled=not st.session_state.evidence,
            use_container_width=True,
        )

    if run:
        try:
            with st.spinner("RootIQ is analyzing the evidence and testing its hypotheses..."):
                previous = st.session_state.analysis
                result = run_investigation()
                st.session_state.analysis = result
                st.session_state.last_follow_up = result.get("follow_up_questions", [])
                st.session_state.history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "analysis": result,
                    }
                )
            st.success("Investigation updated. RootIQ has reassessed the hypothesis against the current evidence.")
        except Exception as exc:
            st.error(f"Investigation failed: {exc}")

    a = st.session_state.analysis
    if a:
        st.divider()
        st.markdown(f"### {a.get('incident_summary', 'Incident')}")
        severity = a.get("severity", "UNKNOWN")
        st.metric("Severity", severity)
        st.write(a.get("severity_reason", ""))

        if a.get("hypothesis_change_note"):
            st.info("**Hypothesis update:** " + a["hypothesis_change_note"])

        st.subheader("Root-cause hypotheses")
        for h in a.get("root_cause_hypotheses", []):
            with st.expander(
                f"{h.get('status', 'Hypothesis')} · {h.get('confidence', 'Unknown')} · {h.get('hypothesis', '')}",
                expanded=True,
            ):
                st.write(h.get("reasoning", ""))
                st.write("**Supporting evidence:**")
                for x in h.get("supporting_evidence", []):
                    st.write(f"- {x}")
                if h.get("contradicting_evidence"):
                    st.write("**Contradicting evidence:**")
                    for x in h["contradicting_evidence"]:
                        st.write(f"- {x}")

        st.subheader("Timeline")
        timeline_df = pd.DataFrame(a.get("timeline", []))
        if not timeline_df.empty:
            st.dataframe(timeline_df, use_container_width=True, hide_index=True)

        st.subheader("Impact")
        impact = a.get("impact") or {}
        for category, values in impact.items():
            if values:
                st.markdown(f"**{category.title()}**")
                st.dataframe(pd.DataFrame(values), use_container_width=True, hide_index=True)

        st.subheader("Actions")
        left, right = st.columns(2)
        with left:
            st.markdown("**Corrective actions**")
            for x in a.get("corrective_actions", []):
                st.write(f"- {x}")
        with right:
            st.markdown("**Preventive actions**")
            for x in a.get("preventive_actions", []):
                st.write(f"- {x}")

        st.subheader("Limitations")
        for x in a.get("limitations", []):
            st.write(f"- {x}")

# Follow-up evidence feature
with tabs[2]:
    st.subheader("🔁 RootIQ asks for missing evidence")
    st.write(
        "This is the new iterative investigation loop: RootIQ identifies an evidence gap, "
        "asks you for it, you upload it, and RootIQ updates the hypothesis using the new data."
    )

    a = st.session_state.analysis
    questions = (a or {}).get("follow_up_questions", [])

    if not a:
        st.info("Run the first investigation to let RootIQ identify evidence gaps.")
    elif not questions:
        st.success(
            "RootIQ did not identify a high-value evidence gap from the current evidence. "
            "You can still add more evidence and run an update."
        )
    else:
        for idx, q in enumerate(questions, 1):
            st.markdown(f"### {idx}. {q.get('question', '')}")
            st.write(f"**Why RootIQ needs it:** {q.get('why_needed', '')}")
            st.write(f"**Expected value:** {q.get('expected_value', '')}")

        st.divider()
        followup_uploads = st.file_uploader(
            "Upload the requested evidence",
            type=[
                "csv", "xlsx", "xls", "pdf", "txt", "log", "md", "json",
                "xml", "html", "eml", "docx", "png", "jpg", "jpeg", "webp"
            ],
            accept_multiple_files=True,
            key="followup_uploader",
        )

        if st.button(
            "🔄 Add evidence & update hypothesis",
            type="primary",
            disabled=not followup_uploads,
        ):
            existing_names = {x["name"] for x in st.session_state.evidence}
            added = 0
            for file in followup_uploads:
                if file.name not in existing_names:
                    st.session_state.evidence.append(parse_uploaded_file(file))
                    added += 1

            if added:
                try:
                    with st.spinner(
                        "New evidence added. RootIQ is reassessing the previous hypothesis..."
                    ):
                        result = run_investigation()
                        st.session_state.analysis = result
                        st.session_state.last_follow_up = result.get("follow_up_questions", [])
                        st.session_state.history.append(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "analysis": result,
                            }
                        )
                    st.success(
                        "Hypothesis updated. RootIQ compared the new evidence with the previous analysis."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Update failed: {exc}")
            else:
                st.warning("Those filenames are already in the investigation.")

# Chat
with tabs[3]:
    st.subheader("💬 Investigation chatbot")
    st.write("Ask questions about the evidence and current hypothesis.")
    if not st.session_state.analysis:
        st.info("Run RootIQ first.")
    else:
        question = st.chat_input("e.g. What evidence supports the leading root cause?")
        if question:
            st.session_state.history.append({"role": "user", "content": question})
            try:
                with st.spinner("Checking the evidence..."):
                    answer = chat_with_investigation(question)
                st.session_state.history.append({"role": "assistant", "content": answer})
            except Exception as exc:
                st.error(f"Chat failed: {exc}")

        for msg in st.session_state.history:
            if "role" in msg:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

# What-if
with tabs[4]:
    st.subheader("🔮 What-if analysis")
    if not st.session_state.analysis:
        st.info("Run RootIQ first.")
    else:
        what_if_q = st.text_input(
            "What would you like to simulate?",
            placeholder="What if the suspected configuration change had been rolled back 30 minutes earlier?",
        )
        if st.button("Run scenario", disabled=not what_if_q):
            try:
                with st.spinner("Running evidence-grounded scenario analysis..."):
                    st.markdown(run_what_if(what_if_q))
            except Exception as exc:
                st.error(f"What-if analysis failed: {exc}")

# Report
with tabs[5]:
    st.subheader("📄 Investigation report")
    if not st.session_state.analysis:
        st.info("Run RootIQ first.")
    else:
        report = report_markdown()
        st.download_button(
            "Download Markdown report",
            data=report,
            file_name="rootiq_investigation_report.md",
            mime="text/markdown",
        )
        st.markdown(report)
