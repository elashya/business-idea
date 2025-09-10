import json, textwrap, time, random
from typing import List, Dict, Callable
import streamlit as st
from pydantic import BaseModel, Field, ValidationError, conlist, confloat, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential
import pandas as pd

# =======================
# App Settings
# =======================
APP_TITLE = "Startup Idea Copilot – Multi-SME Evaluation"
MAX_WORDS = 500

LITE_LENSES = [
    ("business", "Business Viability"),
    ("finance", "Finance & Monetization"),
    ("technology", "Technology & Infrastructure"),
    ("legal", "Legal & Compliance"),
    ("privacy", "Privacy & Data Protection"),
    ("marketing", "Marketing & Growth"),
    ("operations", "Operations & Scalability"),
]
PRO_LENSES = LITE_LENSES + [
    ("consumer", "Consumer Behaviour & UX"),
    ("competition", "Competitive Landscape & Defensibility"),
    ("team", "Team & Execution Feasibility"),
    ("partners", "Strategic Partnerships & Ecosystem Fit"),
    ("ethics", "Ethics & Social Impact"),
    ("branding", "Branding & Positioning"),
    ("support", "Customer Support & Serviceability"),
]

DEFAULT_WEIGHTS = {
    "business": 10, "finance": 10, "technology": 10,
    "legal": 8, "privacy": 8, "marketing": 10, "operations": 8,
    "consumer": 8, "competition": 8, "team": 8, "partners": 6,
    "ethics": 4, "branding": 6, "support": 6,
}

# Short rubric snippets (extend anytime)
RUBRICS = {
    "business": "10: clear problem/solution/TAM; 7–8: decent fit; 5–6: unclear target; ≤4: weak value prop.",
    "finance": "10: strong pricing, CAC/LTV logic; 7–8 plausible; 5–6 thin; ≤4 not viable.",
    "technology": "10: feasible with low risk; 7–8: some unknowns; 5–6: risky; ≤4: not feasible.",
    "legal": "10: low regulatory burden; 7–8: manageable; 5–6: notable hurdles; ≤4: blocked.",
    "privacy": "10: privacy-by-design; 7–8: gaps; 5–6: risks; ≤4: unacceptable.",
    "marketing": "10: clear ICP/channels; 7–8: okay; 5–6: vague; ≤4: weak go-to-market.",
    "operations": "10: scalable ops; 7–8: manageable; 5–6: fragile; ≤4: bottlenecks.",
    "consumer": "10: strong adoption hooks; 7–8: okay UX; 5–6: friction; ≤4: low adoption.",
    "competition": "10: defensible moat; 7–8: differentiation; 5–6: crowded; ≤4: copyable.",
    "team": "10: complete team; 7–8: minor gaps; 5–6: key gaps; ≤4: weak execution.",
    "partners": "10: obvious alliances; 7–8: some; 5–6: unclear; ≤4: isolated.",
    "ethics": "10: positive impact; 7–8: neutral; 5–6: concerns; ≤4: harmful.",
    "branding": "10: crisp positioning; 7–8: decent; 5–6: bland; ≤4: confusing.",
    "support": "10: self-serve, low cost; 7–8: manageable; 5–6: costly; ≤4: unsustainable.",
}

# Few-shot anchors (add more over time)
FEW_SHOTS = {
    "finance": {
        "idea": "A SaaS tool for freelancers that tracks invoices and auto-chases late payments.",
        "json": {
            "lens_key": "finance",
            "score": 7.5,
            "strengths": ["Recurring revenue", "Clear pricing tiers", "Large freelancer market"],
            "weaknesses": ["Unclear CAC channels", "Limited enterprise upsell story"],
            "risks": ["Churn risk if AR automation under-delivers", "Competitive price pressure"],
            "mitigations": ["Annual plan discount", "Partnerships with marketplaces", "Strong onboarding to reduce churn"]
        }
    },
    "technology": {
        "idea": "A mobile-first habit tracking app using push nudges and lightweight ML for personalization.",
        "json": {
            "lens_key": "technology",
            "score": 7.0,
            "strengths": ["Simple stack feasible", "Low infra needs", "Fast iteration cycles"],
            "weaknesses": ["Model performance may be limited by sparse data"],
            "risks": ["Cold-start personalization", "Vendor lock-in for notifications"],
            "mitigations": ["Hybrid rules+ML", "Event telemetry and A/B tests", "Abstract notification provider"]
        }
    },
    "legal": {
        "idea": "An AI résumé enhancer that rewrites resumes and drafts cover letters for job seekers.",
        "json": {
            "lens_key": "legal",
            "score": 7.0,
            "strengths": ["Low regulatory burden", "Clear terms of service"],
            "weaknesses": ["Potential IP ownership confusion on generated text"],
            "risks": ["Misrepresentation concerns", "Jurisdictional employment law variance"],
            "mitigations": ["Clear disclaimers", "User ownership of outputs", "Localized compliance guidance"]
        }
    }
}

# Prompt A/B variants per lens
def prompt_a(rubric: str, lens_name: str, lens_key: str) -> str:
    return f"""You are a senior SME specializing in {lens_name}.
Evaluate the startup idea (<=500 words) with rigor.
Return STRICT JSON only:
{{
  "lens_key": "{lens_key}",
  "score": number 0..10,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "risks": ["..."],
  "mitigations": ["..."]
}}
Scoring rubric:
{rubric}
Be concise, specific, and actionable. No prose outside JSON."""

def prompt_b(rubric: str, lens_name: str, lens_key: str) -> str:
    return f"""Act as an investor-level {lens_name} expert.
Score precisely and justify concisely. JSON ONLY:
{{
  "lens_key": "{lens_key}",
  "score": number 0..10,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "risks": ["..."],
  "mitigations": ["..."]
}}
Rubric guide:
{rubric}
Avoid generic advice. No extra text."""

PROMPT_VARIANTS: Dict[str, List[Callable[[str, str, str], str]]] = {
    # defaults for all lenses
    "*": [prompt_a, prompt_b],
    # you could override per-lens if needed, e.g.:
    # "finance": [prompt_b, prompt_a],
}

# =======================
# OpenAI client
# =======================
def get_openai_client():
    try:
        from openai import OpenAI
        return OpenAI()  # expects OPENAI_API_KEY in Streamlit secrets
    except Exception as e:
        raise RuntimeError(
            "OpenAI client not available. Set OPENAI_API_KEY in Streamlit secrets and add `openai` to requirements."
        ) from e

# =======================
# Schemas & Validation
# =======================
class LensOutput(BaseModel):
    lens_key: str
    score: confloat(ge=0, le=10)
    strengths: conlist(str, min_items=1, max_items=8)
    weaknesses: conlist(str, min_items=1, max_items=8)
    risks: conlist(str, min_items=1, max_items=8)
    mitigations: conlist(str, min_items=1, max_items=8)

    @field_validator("strengths", "weaknesses", "risks", "mitigations")
    @classmethod
    def no_empty_or_na(cls, v):
        cleaned = [s.strip() for s in v if isinstance(s, str) and s.strip().lower() not in {"n/a", "none", "-", ""}]
        if not cleaned:
            raise ValueError("Lists must contain at least one meaningful item.")
        return cleaned

def sanitize_output(data: dict) -> dict:
    data["score"] = float(max(0, min(10, data.get("score", 0))))
    for k in ("strengths", "weaknesses", "risks", "mitigations"):
        seq = data.get(k, [])
        data[k] = [s for s in seq if isinstance(s, str) and s.strip()][:8]
        if not data[k]:
            data[k] = ["(no item)"]
    if not data.get("lens_key"): data["lens_key"] = "unknown"
    return data

# =======================
# Prompts
# =======================
def lens_system_prompt(lens_key: str, lens_name: str, variant_idx: int = 0) -> str:
    rubric = RUBRICS[lens_key]
    builders = PROMPT_VARIANTS.get(lens_key, PROMPT_VARIANTS["*"])
    builder = builders[variant_idx % len(builders)]
    return builder(rubric, lens_name, lens_key)

def exec_summary_prompt() -> str:
    return """You are an analyst. Given multiple JSON lens evaluations, write a 120–180 word executive summary:
- Top 3 strengths and top 3 risks
- The 2 most critical mitigations
- Neutral, concise tone
Return plain text only."""

# =======================
# LLM Calls (with repair & few-shot)
# =======================
def _build_messages_for_lens(lens_key: str, lens_name: str, idea_text: str, variant_idx: int):
    msgs = [{"role": "system", "content": lens_system_prompt(lens_key, lens_name, variant_idx)}]
    if lens_key in FEW_SHOTS:
        ex = FEW_SHOTS[lens_key]
        msgs += [
            {"role": "user", "content": "EXAMPLE_IDEA:\n" + ex["idea"]},
            {"role": "assistant", "content": json.dumps(ex["json"], ensure_ascii=False)}
        ]
    msgs.append({"role": "user", "content": "STARTUP_IDEA:\n" + idea_text.strip()})
    return msgs

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.7, min=0.5, max=2))
def call_lens(client, lens_key: str, lens_name: str, idea_text: str, variant_idx: int, seed: int | None):
    msgs = _build_messages_for_lens(lens_key, lens_name, idea_text, variant_idx)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=msgs,
        response_format={"type": "json_object"},
        **({"seed": seed} if seed is not None else {})
    )
    raw = resp.choices[0].message.content
    data = json.loads(raw)
    try:
        LensOutput(**data)
    except ValidationError as ve:
        # Try repair
        data = repair_lens_json(client, lens_key, lens_name, idea_text, data, str(ve), variant_idx, seed)
        LensOutput(**data)
    return sanitize_output(data)

def repair_lens_json(client, lens_key, lens_name, idea_text, bad_json, validation_msg, variant_idx: int, seed: int | None):
    repair_inst = f"""Your previous JSON for lens '{lens_key}' failed validation.
Validation error: {validation_msg}
Return corrected JSON ONLY. Keep the same schema and constraints."""
    msgs = [
        {"role": "system", "content": lens_system_prompt(lens_key, lens_name, variant_idx)},
        {"role": "user", "content": "STARTUP_IDEA:\n" + idea_text.strip()},
        {"role": "user", "content": "PREVIOUS_JSON:\n" + json.dumps(bad_json, ensure_ascii=False)},
        {"role": "user", "content": repair_inst},
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=msgs,
        response_format={"type": "json_object"},
        **({"seed": seed} if seed is not None else {})
    )
    return json.loads(resp.choices[0].message.content)

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.7, min=0.5, max=2))
def call_summary(client, lenses_json: List[Dict], seed: int | None):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": exec_summary_prompt()},
            {"role": "user", "content": json.dumps(lenses_json, ensure_ascii=False)},
        ],
        **({"seed": seed} if seed is not None else {})
    )
    return resp.choices[0].message.content.strip()

# =======================
# Scoring & Tables
# =======================
def aggregate_score(lenses: List[Dict], weights: Dict[str, float]) -> float:
    active = {k: v for k, v in weights.items() if any(l["lens_key"] == k for l in lenses)}
    total_w = sum(active.values()) or 1.0
    s = 0.0
    for l in lenses:
        w = active.get(l["lens_key"], 0.0)
        s += (float(l["score"]) / 10.0) * w * (100.0 / total_w)
    return round(s, 1)

def to_dataframe(lenses: List[Dict]) -> pd.DataFrame:
    rows = []
    for l in lenses:
        rows.append({
            "Lens": l["lens_key"],
            "Score (0-10)": l["score"],
            "Top Strength": l["strengths"][0] if l["strengths"] else "",
            "Top Risk": l["risks"][0] if l["risks"] else "",
            "Top Mitigation": l["mitigations"][0] if l["mitigations"] else "",
        })
    return pd.DataFrame(rows).sort_values("Lens")

# =======================
# UI
# =======================
st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")
st.title(APP_TITLE)
st.caption("Paste your idea (≤500 words). We return multi-SME scores + a concise summary. Nothing is stored unless you download.")

col1, col2 = st.columns([2, 1])
with col1:
    idea_text = st.text_area(
        "Your idea (≤500 words)",
        height=240,
        placeholder="Describe the problem, solution, user, market, monetization, and tech approach...",
    )
    word_count = len(idea_text.split())
    st.write(f"**Word count:** {word_count} / {MAX_WORDS}")

with col2:
    mode = st.radio("Evaluation mode", ["Lite (7 lenses)", "Pro (14 lenses)"], index=0)
    lenses = LITE_LENSES if mode.startswith("Lite") else PRO_LENSES
    variant_choice = st.selectbox("Prompt variant", ["A", "B"], index=0)
    seed_opt = st.checkbox("Set seed (reproducible)")
    seed_val = st.number_input("Seed value", value=42, step=1, disabled=not seed_opt)
    privacy = st.checkbox("Process & delete (do not retain input)", value=True)
    run_btn = st.button("Run Evaluation", type="primary", use_container_width=True)

# Sidebar: adjust weights
with st.sidebar:
    st.subheader("Weights (optional)")
    weights = {}
    for key, name in lenses:
        weights[key] = st.slider(name, 0, 12, DEFAULT_WEIGHTS.get(key, 8), 1)
    st.caption("Weights are normalized to 100 in the final score.")

# =======================
# Action
# =======================
if run_btn:
    if not idea_text.strip():
        st.error("Please paste your idea first.")
        st.stop()
    if word_count > MAX_WORDS:
        st.error(f"Please reduce your idea to ≤ {MAX_WORDS} words.")
        st.stop()

    with st.spinner("Evaluating across SME lenses..."):
        client = get_openai_client()
        results = []
        start = time.time()
        v_idx = 0 if variant_choice == "A" else 1
        seed = int(seed_val) if seed_opt else None

        for k, name in lenses:
            try:
                out = call_lens(client, k, name, idea_text, v_idx, seed)
                results.append(out)
            except Exception as e:
                # graceful degradation
                results.append({
                    "lens_key": k, "score": 0.0,
                    "strengths": [],
                    "weaknesses": [f"Evaluation failed: {str(e)}"],
                    "risks": ["Model output invalid or provider error."],
                    "mitigations": ["Retry later or adjust description for clarity."]
                })

        overall = aggregate_score(results, weights)
        try:
            summary = call_summary(client, results, seed)
        except Exception:
            summary = "Summary unavailable due to provider error. Please review lens tables."

        elapsed = time.time() - start

    st.success(f"Done in {elapsed:.1f}s. Overall Score: **{overall}/100**")
    st.markdown("### Executive Summary")
    st.write(summary)

    st.markdown("### Lens Breakdown")
    df = to_dataframe(results)
    st.dataframe(df, use_container_width=True)

    st.markdown("### Full JSON")
    full = {
        "mode": "pro" if len(lenses) > 7 else "lite",
        "overall_score": overall,
        "weights": {k: weights[k] for k, _ in lenses},
        "lenses": results,
        "executive_summary": summary,
        "rubric_version": "v1.1.0",
        "prompt_variant": variant_choice,
        "seed_used": seed,
        "privacy_process_and_delete": privacy,
    }
    st.code(json.dumps(full, ensure_ascii=False, indent=2), language="json")

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(full, ensure_ascii=False).encode("utf-8"),
        file_name="evaluation_report.json",
        mime="application/json",
        use_container_width=True,
    )

    # Privacy: do not persist input by default
    if privacy:
        del idea_text
