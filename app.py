import json, time, os, hashlib, datetime
from typing import List, Dict, Callable

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from pydantic import BaseModel, ValidationError, conlist, confloat, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

# =========================================================
# App metadata / schema version
# =========================================================
APP_TITLE = "Startup Idea Copilot – Multi-SME Evaluation"
MAX_WORDS = 300
SCHEMA_VERSION = 3  # includes score_rationale

# =========================================================
# Lenses & Weights
# =========================================================
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

# =========================================================
# Rubrics (tight + lens-specific)
# =========================================================
RUBRICS = {
    "business": "10: clear problem-solution fit, defined ICP, credible TAM/SAM, sharp value prop; 7–8: decent clarity; 5–6: vague ICP/value prop; ≤4: no business case.",
    "finance": "10: pricing, CAC/LTV, margins, path to profit coherent; 7–8: plausible; 5–6: thin monetization or unclear unit economics; ≤4: non-viable. Finance only.",
    "technology": "10: feasible architecture; addresses security, reliability, scalability, integrations & data; 7–8: feasible w/ unknowns; 5–6: notable reliability/security risks; ≤4: not feasible. Tech only.",
    "legal": "10: low/clear regulatory burden with plan; 7–8: manageable; 5–6: notable hurdles; ≤4: blocked. Legal only.",
    "privacy": "10: privacy-by-design, minimization, rights, transfer controls; 7–8: small gaps; 5–6: material risks; ≤4: unacceptable. Privacy only.",
    "marketing": "10: clear ICP, messages, channels with early GTM; 7–8: decent; 5–6: vague; ≤4: weak. Marketing only.",
    "operations": "10: scalable processes, SLAs, supply/logistics, support model; 7–8: workable; 5–6: fragile; ≤4: bottlenecks. Ops only.",
    "consumer": "10: strong adoption hooks, low friction, trust; 7–8: okay; 5–6: friction; ≤4: unlikely to adopt. Consumer only.",
    "competition": "10: defensible moat (switching costs, network, IP); 7–8: differentiation; 5–6: crowded; ≤4: copyable. Competition only.",
    "team": "10: team covers domain/tech/GTM; 7–8: minor gaps; 5–6: key gaps; ≤4: weak execution readiness. Team only.",
    "partners": "10: obvious alliances with incentives; 7–8: some; 5–6: unclear; ≤4: isolated. Partners only.",
    "ethics": "10: positive externalities, low misuse risk; 7–8: neutral; 5–6: concerns; ≤4: harmful. Ethics only.",
    "branding": "10: crisp positioning, memorable identity; 7–8: decent; 5–6: bland; ≤4: confusing. Branding only.",
    "support": "10: self-serve UX, low-cost support, clear escalation; 7–8: manageable; 5–6: costly; ≤4: unsustainable. Support only.",
}

# Allowed/Disallowed topics for QA auditor
LENS_TOPICS = {
    "business": {"allowed": ["problem-solution fit","ICP","TAM/SAM","value proposition","business model"], "disallowed": ["encryption","data retention","regulatory fines","database choice"]},
    "finance": {"allowed": ["pricing","CAC","LTV","margins","payback","revenue streams","cost structure"], "disallowed": ["encryption","UX flow","IoT hardware","GDPR rights"]},
    "technology": {"allowed": ["architecture","APIs","data model","security","reliability","scalability","latency","cloud","integrations"], "disallowed": ["market demand","pricing","insurance","contracts","brand message"]},
    "legal": {"allowed": ["contracts","terms","IP","licensing","consumer law","sector regulation","liability"], "disallowed": ["architecture","pricing tiers","ad channels"]},
    "privacy": {"allowed": ["data categories","minimization","retention","consent","DSAR","SCCs","DPIA","data residency"], "disallowed": ["pricing","market size","cloud SLA (unless data related)"]},
    "marketing": {"allowed": ["ICP","positioning","messaging","channels","funnel","growth loops"], "disallowed": ["encryption","DPIA","DB sharding"]},
    "operations": {"allowed": ["process","SLA","supply chain","support model","capacity","QA","vendor mgmt"], "disallowed": ["TAM","pricing","UI color palette"]},
    "consumer": {"allowed": ["trust","friction","adoption triggers","habits","social proof","loss aversion"], "disallowed": ["SCCs","IP transfer","API gateways"]},
    "competition": {"allowed": ["rivals","substitutes","switching costs","network effects","moat"], "disallowed": ["serverless vs VM","cookie banners"]},
    "team": {"allowed": ["skills","experience","hiring plan","advisors"], "disallowed": ["TLS versions","CPC bids"]},
    "partners": {"allowed": ["channels","alliances","platform integrations","institutions"], "disallowed": ["memory leaks","A/B test plan"]},
    "ethics": {"allowed": ["bias","harm","fairness","misuse","externalities","safeguards"], "disallowed": ["pricing tactics","JS framework choice"]},
    "branding": {"allowed": ["positioning","identity","category","promise","naming"], "disallowed": ["DB index","GDPR transfer tool"]},
    "support": {"allowed": ["docs","in-product help","SLA","staffing","ticketing","deflection"], "disallowed": ["market thesis","OCI tenancy"]},
}

# Few-shots (short set with score_rationale anchors)
FEW_SHOTS = {
    "finance": {
        "idea": "A SaaS tool for freelancers that tracks invoices and auto-chases late payments.",
        "json": {
            "lens_key": "finance",
            "score": 7.5,
            "score_rationale": [
                "Recurring revenue with clear tiering",
                "Large addressable base supports volume",
                "CAC and churn risks temper upside"
            ],
            "strengths": ["Recurring revenue","Clear pricing tiers","Large freelancer market"],
            "weaknesses": ["Unclear CAC channels","Limited enterprise upsell story"],
            "risks": ["Churn if AR under-delivers","Price pressure"],
            "mitigations": ["Annual plan discount","Marketplace partnerships","Onboarding to reduce churn"]
        }
    },
    "technology": {
        "idea": "A mobile-first habit tracker using push nudges and lightweight ML personalization.",
        "json": {
            "lens_key": "technology",
            "score": 7.0,
            "score_rationale": [
                "Commodity stack is feasible",
                "Personalization limited by sparse data",
                "Lock-in manageable via abstraction"
            ],
            "strengths": ["Simple stack","Low infra needs","Fast iterations"],
            "weaknesses": ["Sparse data may limit ML"],
            "risks": ["Cold start","Vendor lock-in"],
            "mitigations": ["Hybrid rules+ML","A/B tests","Abstract provider"]
        }
    }
}

# =========================================================
# Prompt variants (A/B)
# =========================================================
def prompt_a(rubric: str, lens_name: str, lens_key: str, allowed: List[str], disallowed: List[str]) -> str:
    return (
        f"You are a senior SME for {lens_name}.\n"
        f"Evaluate ONLY through the {lens_name} lens.\n"
        f"Allowed topics: {', '.join(allowed)}.\n"
        f"Disallowed topics: {', '.join(disallowed)}.\n"
        "Return STRICT JSON only:\n"
        "{\n"
        f'  "lens_key": "{lens_key}",\n'
        '  "score": number 0..10,\n'
        '  "score_rationale": ["2-4 concise, lens-specific reasons grounded in the idea text"],\n'
        '  "strengths": ["..."],\n'
        '  "weaknesses": ["..."],\n'
        '  "risks": ["..."],\n'
        '  "mitigations": ["..."]\n'
        "}\n"
        f"Scoring rubric: {rubric}\n"
        "If any item is not lens-specific, replace it before returning JSON."
    )

def prompt_b(rubric: str, lens_name: str, lens_key: str, allowed: List[str], disallowed: List[str]) -> str:
    return (
        f"Act as an investor-level {lens_name} expert. JSON ONLY.\n"
        f"Allowed topics: {', '.join(allowed)}.\n"
        f"Disallowed topics: {', '.join(disallowed)}.\n"
        "{\n"
        f'  "lens_key": "{lens_key}", "score": number 0..10,\n'
        '  "score_rationale": ["2-4 concise, lens-specific reasons"],\n'
        '  "strengths": ["..."], "weaknesses": ["..."], "risks": ["..."], "mitigations": ["..."]\n'
        "}\n"
        f"Rubric: {rubric}\n"
        "Remove or rewrite any off-topic bullet."
    )

PROMPT_VARIANTS: Dict[str, List[Callable[[str, str, str, List[str], List[str]], str]]] = {
    "*": [prompt_a, prompt_b],
}

# =========================================================
# Secrets: OpenAI + PIN Gate
# =========================================================
def get_openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI package not installed. Check requirements.txt") from e
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it in Streamlit Cloud → App → Settings → Secrets.")
    return OpenAI(api_key=api_key)

def _get_pin_secrets():
    pin_plain = st.secrets.get("APP_PIN") or os.getenv("APP_PIN")
    pin_hash = st.secrets.get("APP_PIN_SHA256") or os.getenv("APP_PIN_SHA256")
    return pin_plain, pin_hash

def require_pin():
    pin_plain, pin_hash = _get_pin_secrets()
    if not pin_plain and not pin_hash:
        return True
    if st.session_state.get("authed"):
        with st.sidebar:
            st.success("🔒 PIN verified")
            if st.button("Log out"):
                st.session_state.clear()
                st.rerun()
        return True
    attempts = st.session_state.get("pin_attempts", 0)
    if attempts >= 5:
        st.error("Too many incorrect attempts. Please refresh and try again later.")
        st.stop()
    st.title("🔐 Enter Access PIN")
    with st.form("pin_form"):
        user_pin = st.text_input("PIN", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        ok = False
        if pin_plain and user_pin == str(pin_plain):
            ok = True
        elif pin_hash:
            import hashlib as _hl
            h = _hl.sha256(user_pin.encode("utf-8")).hexdigest()
            if h == pin_hash:
                ok = True
        if ok:
            st.session_state["authed"] = True
            st.session_state["pin_attempts"] = 0
            st.rerun()
        else:
            st.session_state["pin_attempts"] = attempts + 1
            st.error("Incorrect PIN.")
            st.stop()
    else:
        st.stop()

# =========================================================
# Pydantic schema (v2) with score_rationale
# =========================================================
class LensOutput(BaseModel):
    lens_key: str
    score: confloat(ge=0, le=10)
    score_rationale: conlist(str, min_length=1, max_length=4)
    strengths: conlist(str, min_length=1, max_length=8)
    weaknesses: conlist(str, min_length=1, max_length=8)
    risks: conlist(str, min_length=1, max_length=8)
    mitigations: conlist(str, min_length=1, max_length=8)

    @field_validator("strengths", "weaknesses", "risks", "mitigations", "score_rationale")
    @classmethod
    def no_empty_or_na(cls, v):
        cleaned = [s.strip() for s in v if isinstance(s, str) and s.strip().lower() not in {"n/a","none","-",""}]
        if not cleaned:
            raise ValueError("Lists must contain at least one meaningful item.")
        return cleaned

def sanitize_output(data: dict) -> dict:
    data["score"] = float(max(0, min(10, data.get("score", 0))))
    for k in ("strengths","weaknesses","risks","mitigations","score_rationale"):
        seq = data.get(k, [])
        data[k] = [s for s in seq if isinstance(s, str) and s.strip()][:8]
        if not data[k]:
            data[k] = ["(no item)"]
    if not data.get("lens_key"):
        data["lens_key"] = "unknown"
    return data

# =========================================================
# Prompt builders / QA auditor
# =========================================================
def lens_system_prompt(lens_key: str, lens_name: str, variant_idx: int = 0) -> str:
    rubric = RUBRICS[lens_key]
    topics = LENS_TOPICS[lens_key]
    builders = PROMPT_VARIANTS.get(lens_key, PROMPT_VARIANTS["*"])
    builder = builders[variant_idx % len(builders)]
    return builder(rubric, lens_name, lens_key, topics["allowed"], topics["disallowed"])

def exec_summary_prompt() -> str:
    return ("You are an analyst. Given multiple JSON lens evaluations, write a 120–180 word executive summary:\n"
            "- Top 3 strengths and top 3 risks\n- The 2 most critical mitigations\n- Neutral, concise tone\n"
            "Return plain text only.")

def lens_auditor_prompt(lens_key: str, lens_name: str) -> str:
    topics = LENS_TOPICS[lens_key]
    return (f"You are a QA reviewer for the {lens_name} lens.\n"
            f"Allowed topics: {', '.join(topics['allowed'])}.\n"
            f"Disallowed topics: {', '.join(topics['disallowed'])}.\n"
            "Given the JSON below, REWRITE any off-topic bullet to become lens-relevant, or remove it. "
            "Return STRICT JSON with the same schema.")

# =========================================================
# LLM calls (with repair + few-shot + auditor)
# =========================================================
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
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=_build_messages_for_lens(lens_key, lens_name, idea_text, variant_idx),
        response_format={"type": "json_object"},
        **({"seed": seed} if seed is not None else {})
    )
    data = json.loads(resp.choices[0].message.content)
    try:
        LensOutput(**data)
    except ValidationError as ve:
        data = repair_lens_json(client, lens_key, lens_name, idea_text, data, str(ve), variant_idx, seed)
        LensOutput(**data)
    return sanitize_output(data)

def repair_lens_json(client, lens_key, lens_name, idea_text, bad_json, validation_msg, variant_idx: int, seed: int | None):
    msgs = [
        {"role": "system", "content": lens_system_prompt(lens_key, lens_name, variant_idx)},
        {"role": "user", "content": "STARTUP_IDEA:\n" + idea_text.strip()},
        {"role": "user", "content": "PREVIOUS_JSON:\n" + json.dumps(bad_json, ensure_ascii=False)},
        {"role": "user", "content": f"Your previous JSON for '{lens_key}' failed validation: {validation_msg}\nReturn corrected JSON ONLY, same schema."},
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=msgs,
        response_format={"type": "json_object"},
        **({"seed": seed} if seed is not None else {})
    )
    return json.loads(resp.choices[0].message.content)

def audit_lens_relevance(client, lens_key: str, lens_name: str, data: dict, seed: int | None) -> dict:
    msgs = [
        {"role": "system", "content": lens_auditor_prompt(lens_key, lens_name)},
        {"role": "user", "content": json.dumps(data, ensure_ascii=False)}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=msgs,
        response_format={"type": "json_object"},
        **({"seed": seed} if seed is not None else {})
    )
    revised = json.loads(resp.choices[0].message.content)
    try:
        LensOutput(**revised)
        return sanitize_output(revised)
    except ValidationError:
        return sanitize_output(data)

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

# =========================================================
# Scoring / Printable HTML / Fingerprint
# =========================================================
def aggregate_score(lenses: List[Dict], weights: Dict[str, float]) -> float:
    active = {k: v for k, v in weights.items() if any(l["lens_key"] == k for l in lenses)}
    total_w = sum(active.values()) or 1.0
    s = 0.0
    for l in lenses:
        w = active.get(l["lens_key"], 0.0)
        s += (float(l["score"]) / 10.0) * w * (100.0 / total_w)
    return round(s, 1)

def to_dataframe_detailed(lenses: List[Dict], weights: Dict[str, float]) -> pd.DataFrame:
    active_w = {l["lens_key"]: float(weights.get(l["lens_key"], 0)) for l in lenses}
    total_w = sum(active_w.values()) or 1.0
    rows = []
    for l in lenses:
        k = l["lens_key"]; w = active_w.get(k, 0.0)
        contrib = (float(l["score"])/10.0) * w * (100.0/total_w)
        rows.append({
            "Lens": k.capitalize(),
            "Score (0–10)": round(float(l["score"]), 2),
            "Weight": int(w),
            "Contribution (%)": round(contrib, 1),
            "Why this score": (l.get("score_rationale") or [""])[0],
            "Top Strength": l["strengths"][0] if l["strengths"] else "",
            "Top Risk": l["risks"][0] if l["risks"] else "",
            "Top Mitigation": l["mitigations"][0] if l["mitigations"] else "",
        })
    return pd.DataFrame(rows).sort_values("Lens").reset_index(drop=True)

def config_fingerprint(lenses, weights, variant_choice, rubrics) -> str:
    payload = {
        "lenses": [k for k, _ in lenses],
        "weights": {k: weights[k] for k, _ in lenses},
        "variant": variant_choice,
        "rubrics": {k: rubrics[k] for k, _ in lenses},
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return h[:8]

def build_printable_html(full_json: dict, df: pd.DataFrame, summary: str, overall: float, cfg_fp: str) -> str:
    table_html = df.to_html(index=False, border=0, escape=True)
    safe_json = json.dumps(full_json, ensure_ascii=False, indent=2)
    css = """
    <style>
      body { font-family: Inter, Arial, sans-serif; margin: 32px; }
      h1,h2,h3 { margin: 0 0 8px; }
      .meta { color: #555; margin-bottom: 16px; }
      table { border-collapse: collapse; width: 100%; margin: 12px 0; }
      th, td { border: 1px solid #ddd; padding: 8px; font-size: 12.5px; white-space: normal; word-wrap: break-word; }
      pre { background: #fafafa; border: 1px solid #eee; padding: 12px; overflow: auto; }
      @media print { .no-print { display: none !important; } body { margin: 8mm; } }
    </style>
    """
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Startup Idea Report</title>{css}</head>
<body onload="window.print()">
  <h1>Startup Idea Evaluation Report</h1>
  <div class="meta">Overall Score: <b>{overall}/100</b> • Config: {cfg_fp} • Schema v{SCHEMA_VERSION}</div>
  <h2>Executive Summary</h2>
  <p>{summary}</p>
  <h2>Lens Breakdown</h2>
  {table_html}
  <h2>Full JSON</h2>
  <pre>{safe_json}</pre>
  <button class="no-print" onclick="window.print()">Print</button>
</body></html>"""

# =========================================================
# UI
# =========================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")
require_pin()

st.title(APP_TITLE)
st.caption("Paste your idea (≤300 words). We return multi-SME scores + a concise summary. Nothing is stored unless you download.")

col1, col2 = st.columns([2, 1])
with col1:
    idea_text = st.text_area(
        "Your idea (≤300 words)",
        height=240,
        placeholder="Describe the problem, solution, user, market, monetization, and tech approach...",
    )
    word_count = len(idea_text.split())
    st.write(f"**Word count:** {word_count} / {MAX_WORDS}")

with col2:
    mode = st.radio("Evaluation mode", ["Lite (7 lenses)", "Pro (14 lenses)"], index=0)
    lenses = LITE_LENSES if mode.startswith("Lite") else PRO_LENSES
    variant_choice = st.selectbox("Prompt variant", ["A", "B"], index=0)
    strict_qc = st.checkbox("Strict lens relevance (extra QA)", value=True)
    seed_opt = st.checkbox("Set seed (reproducible)")
    seed_val = st.number_input("Seed value", value=42, step=1, disabled=not seed_opt)
    privacy = st.checkbox("Process & delete (do not retain input)", value=True)
    run_btn = st.button("Run Evaluation", type="primary", use_container_width=True)

# Sidebar: weights that should live-update overall score
with st.sidebar:
    st.subheader("Weights (optional)")
    weights = {}
    for key, name in lenses:
        # If we have a previous run with this lens, prefill its weight; else default
        default_val = DEFAULT_WEIGHTS.get(key, 8)
        weights[key] = st.slider(name, 0, 12, default_val, 1)
    st.caption("Weights affect the **overall 100** only; they do **not** change per-lens scores.")

# =========================================================
# Run or live-reweight logic (session-state caching)
# =========================================================
def lens_keys(lst): return [k for k, _ in lst]

def store_results(_results, _summary, _lenses, _variant, _seed):
    st.session_state["last_results"] = _results
    st.session_state["last_summary"] = _summary
    st.session_state["last_lens_keys"] = lens_keys(_lenses)
    st.session_state["last_variant"] = _variant
    st.session_state["last_seed"] = int(_seed) if _seed is not None else None
    st.session_state["last_model"] = "gpt-4o-mini"
    st.session_state["last_timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"

def have_cached_results_for_current_mode() -> bool:
    return (
        "last_results" in st.session_state
        and st.session_state.get("last_lens_keys") == lens_keys(lenses)
    )

def render_results(results, summary, lenses_for_run, variant_used, seed_used):
    # Executive Summary at TOP
    overall_now = aggregate_score(results, {k: weights[k] for k, _ in lenses_for_run})
    cfg_fp = config_fingerprint(lenses_for_run, weights, variant_used, RUBRICS)

    st.success(f"Overall Score: **{overall_now}/100**")
    st.caption(f"Config fingerprint: `{cfg_fp}` • Schema v{SCHEMA_VERSION}")
    if variant_choice != variant_used or lens_keys(lenses) != lens_keys(lenses_for_run):
        st.info(f"Showing results from previous run • Variant {variant_used} • Mode with {len(lenses_for_run)} lenses. "
                f"Change requires re-run.")

    st.markdown("### Executive Summary")
    st.write(summary)

    # Bubbles (expanders) — no weight/% in header
    st.markdown("### Lens Details")
    # order by lens name for consistent UX
    sorted_results = sorted(results, key=lambda l: l["lens_key"])
    for l in sorted_results:
        with st.expander(f"{l['lens_key'].capitalize()} — {l['score']}/10"):
            st.markdown("**Why this score**")
            for r in l.get("score_rationale", []):
                st.markdown(f"- {r}")

            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Strengths**")
                for s in l.get("strengths", [])[:6]: st.markdown(f"- {s}")
                st.markdown("**Mitigations**")
                for m in l.get("mitigations", [])[:6]: st.markdown(f"- {m}")
            with colB:
                st.markdown("**Weaknesses**")
                for ww in l.get("weaknesses", [])[:6]: st.markdown(f"- {ww}")
                st.markdown("**Risks**")
                for rr in l.get("risks", [])[:6]: st.markdown(f"- {rr}")

    # Download / Print
    st.markdown("### Full JSON")
    full = {
        "schema_version": SCHEMA_VERSION,
        "mode": "pro" if len(lenses_for_run) > 7 else "lite",
        "overall_score": overall_now,
        "weights": {k: weights[k] for k, _ in lenses_for_run},
        "lenses": results,
        "executive_summary": summary,
        "prompt_variant": variant_used,
        "strict_qc": strict_qc,
        "seed_used": seed_used,
        "model": st.session_state.get("last_model", "gpt-4o-mini"),
        "config_fingerprint": cfg_fp,
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
    df_print = to_dataframe_detailed(results, {k: weights[k] for k, _ in lenses_for_run})
    printable_html = build_printable_html(full, df_print, summary, overall_now, cfg_fp)
    st.download_button(
        "Download printable report (HTML)",
        data=printable_html.encode("utf-8"),
        file_name="evaluation_report.html",
        mime="text/html",
        use_container_width=True,
    )
    if st.button("🖨️ Print now", use_container_width=True):
        components.html(printable_html, height=0, width=0)

# When user clicks Run → call LLMs and cache results
if run_btn:
    if not idea_text.strip():
        st.error("Please paste your idea first."); st.stop()
    if word_count > MAX_WORDS:
        st.error(f"Please reduce your idea to ≤ {MAX_WORDS} words."); st.stop()

    with st.spinner("Evaluating across SME lenses..."):
        client = get_openai_client()
        results = []
        start = time.time()
        v_idx = 0 if variant_choice == "A" else 1
        seed = int(seed_val) if seed_opt else None

        for k, name in lenses:
            try:
                out = call_lens(client, k, name, idea_text, v_idx, seed)
                if strict_qc:
                    out = audit_lens_relevance(client, k, name, out, seed)
                results.append(out)
            except Exception as e:
                results.append({
                    "lens_key": k, "score": 0.0,
                    "score_rationale": ["Evaluation failed"],
                    "strengths": [],
                    "weaknesses": [f"Evaluation failed: {str(e)}"],
                    "risks": ["Model output invalid or provider error."],
                    "mitigations": ["Retry later or adjust description for clarity."]
                })

        try:
            summary = call_summary(client, results, seed)
        except Exception:
            summary = "Summary unavailable due to provider error. Please review lens details."
        elapsed = time.time() - start
        st.toast(f"Evaluation done in {elapsed:.1f}s.", icon="✅")

    # Cache for live reweighting
    store_results(results, summary, lenses, variant_choice, seed)

# If we have cached results for this mode, show them and LIVE reweight with sliders
if have_cached_results_for_current_mode():
    render_results(
        st.session_state["last_results"],
        st.session_state["last_summary"],
        [(k, next(name for (k2, name) in (lenses) if k2 == k)) for k in st.session_state["last_lens_keys"]],
        st.session_state.get("last_variant", "A"),
        st.session_state.get("last_seed"),
    )
else:
    if "last_results" in st.session_state:
        st.info("You changed mode (Lite/Pro) or lens set. Click **Run Evaluation** to regenerate results for this mode.")
