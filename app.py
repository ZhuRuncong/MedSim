"""MedSim — Streamlit front-end (PLAN §6).

Run with:  streamlit run app.py

A clean, academic/clinical UI for the gamified case simulator. All game logic
lives in ``src/engine.py``; this file is presentation + state wiring only.
"""
from __future__ import annotations

import html
import random

import streamlit as st

from src import config, data_loader, engine
from src.store import get_store

st.set_page_config(page_title="MedSim — Clinical Case Simulator", layout="wide")

# --------------------------------------------------------------------------- #
# Styling — light, academic palette; hairline borders; no glow / no glass blur.
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
/* System font stacks — no external CDN, keeping the app fully offline/local-first.
   Georgia provides the academic serif for headings. */
:root {
  --ink: #1b2733;
  --muted: #5f6f7e;
  --line: #e3e8ee;
  --line-strong: #cbd4de;
  --bg: #ffffff;
  --panel: #ffffff;
  --panel-alt: #f7f9fb;
  --accent: #1f4e79;
  --accent-ink: #163a5c;
  --accent-soft: #eef3f8;
  --gain: #2f7d5b;
  --loss: #b23b30;
}
html, body, [class*="css"], .stApp {
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: var(--ink);
}
.stApp { background: var(--bg); }

/* Hide Streamlit chrome for a cleaner product surface */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
header[data-testid="stHeader"] { background: transparent; height: 0; }

section[data-testid="stSidebar"] {
  background: var(--panel-alt);
  border-right: 1px solid var(--line);
}

.serif { font-family: Georgia, 'Times New Roman', serif; }

.wordmark { font-family: Georgia, 'Times New Roman', serif; font-size: 1.4rem; font-weight: 700;
            color: var(--ink); letter-spacing: -.2px; }
.hero-title { font-family: Georgia, 'Times New Roman', serif; font-weight: 600; font-size: 1.75rem;
              color: var(--ink); letter-spacing: -.3px; margin: 0; }
.hero-title .accent { color: var(--accent); }
.subtitle { color: var(--muted); font-size: .9rem; margin-top: 3px; }

.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px 18px;
  margin-bottom: 14px;
}
.card.center { text-align: center; }

.vital { display: inline-block; text-align: center; margin: 6px 20px 4px 0; }
.vital .v { font-size: 1.35rem; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }
.vital .k { font-size: .66rem; color: var(--muted); text-transform: uppercase; letter-spacing: .09em; }

.tag { display: inline-block; padding: 2px 9px; border-radius: 4px; font-size: .68rem;
       font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
       border: 1px solid var(--line-strong); color: var(--muted); background: var(--panel-alt); }
.tag.accent { color: var(--accent); border-color: #c2d3e5; background: var(--accent-soft); }
.tag.alert  { color: var(--loss); border-color: #e6c9c4; background: #fbf1ef; }
.tag.ok     { color: var(--gain); border-color: #bfe0cf; background: #eef7f2; }
.tag.miss   { color: var(--muted); border-color: var(--line); background: var(--panel-alt); text-transform: none; }
.dbrief-row { margin: 7px 0; font-size: .86rem; }
.dbrief-row .lbl { display:inline-block; min-width: 118px; color: var(--muted); font-size:.72rem;
                   text-transform: uppercase; letter-spacing:.06em; vertical-align: top; }
.dbrief-teach { margin: 4px 0 4px 16px; color: var(--ink); font-size:.86rem; line-height:1.5; }
.dbrief-safety { color: var(--loss); font-weight:600; font-size:.86rem; margin:3px 0; }

.chief { font-size: 1.02rem; color: #2b3949; line-height: 1.45; margin: 6px 0 2px; }

.metric { font-size: 2rem; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; line-height: 1; }
.metric-label { font-size: .66rem; color: var(--muted); text-transform: uppercase; letter-spacing: .09em; margin-top: 5px; }

.section-head { font-family: Georgia, 'Times New Roman', serif; font-weight: 600; font-size: 1.2rem;
                color: var(--ink); border-bottom: 1px solid var(--line); padding-bottom: 6px; margin: 6px 0 4px; }

.feed-item { border-left: 3px solid var(--line-strong); padding: 8px 14px; margin: 8px 0;
             background: var(--panel-alt); border-radius: 0 6px 6px 0; }
.feed-item.gain { border-left-color: var(--gain); }
.feed-item.loss, .feed-item.fail { border-left-color: var(--loss); }
.feed-item.success { border-left-color: var(--accent); background: var(--accent-soft); }
.feed-role { font-size: .66rem; text-transform: uppercase; letter-spacing: .09em; color: var(--muted); }
.feed-text { color: var(--ink); margin-top: 3px; line-height: 1.5; }
.delta-gain { color: var(--gain); font-weight: 600; }
.delta-loss { color: var(--loss); font-weight: 600; }
.cite a { color: var(--accent); font-size: .76rem; text-decoration: underline; margin-right: 12px; }
.badge { display: inline-block; font-size: .6rem; font-weight: 700; letter-spacing: .06em;
         padding: 1px 6px; border-radius: 3px; margin-left: 6px; vertical-align: middle; }
.badge.llm { color: #6b3fa0; background: #f1ecfa; border: 1px solid #d9caf0; }
.traj { font-size: .8rem; border-collapse: collapse; width: 100%; }
.traj th { text-align: left; color: var(--muted); font-weight: 600; font-size: .66rem;
           text-transform: uppercase; letter-spacing: .07em; padding: 4px 10px; border-bottom: 1px solid var(--line); }
.traj td { padding: 5px 10px; border-bottom: 1px solid var(--line); color: var(--ink); }
.traj .mono { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; color: var(--accent-ink); }
.arch { font-size: .82rem; line-height: 1.55; color: var(--ink); }
.arch code { background: var(--panel-alt); border: 1px solid var(--line); border-radius: 4px;
             padding: 0 5px; font-size: .78rem; color: var(--accent-ink); }

.stButton>button {
  background: var(--accent); color: #ffffff; font-weight: 500;
  border: 1px solid var(--accent-ink); border-radius: 6px; transition: background .12s ease;
}
.stButton>button:hover { background: var(--accent-ink); color: #ffffff; }
.stButton>button:disabled { background: #aeb9c4; border-color: #aeb9c4; color: #f0f2f5; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Session bootstrap
# --------------------------------------------------------------------------- #
def _init():
    ss = st.session_state
    ss.setdefault("store", get_store())
    ss.setdefault("state", None)
    # Guarded inside data_loader (alongside the registry) so it reloads if a
    # hot-reload wipes module globals — not tracked in session_state, which
    # would survive the reload and wrongly skip repopulation.
    data_loader.ensure_generated_loaded()  # cached AI drugs + cases


_init()
ss = st.session_state


def _do(action: dict):
    if ss.state is None:
        return
    engine.perform_action(ss.state, action, store=ss.store)


def _fmt(v):
    """Render a vital without a trailing '.0'."""
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


def _langgraph_active() -> bool:
    import os
    if os.getenv("MEDSIM_USE_LANGGRAPH", "1") == "0":
        return False
    try:
        from src.agents import graph
        return graph.LANGGRAPH_AVAILABLE
    except Exception:
        return False


def _tags(items, cls):
    return " ".join(f"<span class='tag {cls}'>{html.escape(str(i))}</span>" for i in items)


def _debrief_html(d) -> str:
    verdict = "ok" if d.outcome == "correct" else "alert"
    verdict_txt = "CORRECT" if d.outcome == "correct" else "MISSED"
    llm_badge = "<span class='badge llm'>LLM</span>" if d.llm else ""

    rows = []

    def row(label, inner):
        if inner:
            rows.append(f"<div class='dbrief-row'><span class='lbl'>{label}</span>{inner}</div>")

    row("Tests", _tags(d.tests_hit, "ok") + " " + _tags(d.tests_missed, "miss")
        + " " + _tags(d.tests_low_value, "alert"))
    row("Exams", _tags(d.exams_hit, "ok") + " " + _tags(d.exams_missed, "miss"))
    row("Drugs", _tags(d.drugs_first_line, "ok") + " " + _tags(d.drugs_missed_first_line, "miss")
        + " " + _tags(d.drugs_harmful, "alert"))
    if d.surgery_note:
        row("Surgery", f"<span style='font-size:.86rem'>{html.escape(d.surgery_note)}</span>")
    if d.key_findings:
        row("Key findings", "<span style='font-size:.86rem'>"
            + "; ".join(html.escape(k) for k in d.key_findings) + "</span>")

    teach = "".join(f"<div class='dbrief-teach'>• {html.escape(t)}</div>" for t in d.teaching_points)
    safety = "".join(f"<div class='dbrief-safety'>⚠ {html.escape(s)}</div>" for s in d.safety_flags)
    cite = (f"<div class='cite'><a href='{html.escape(d.citation)}' target='_blank'>guideline</a></div>"
            if d.citation else "")

    return (
        "<div class='card'>"
        "<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap'>"
        f"<div class='section-head' style='border:0;margin:0'>Case debrief — {html.escape(d.true_diagnosis)}</div>"
        f"<div><span class='tag {verdict}'>{verdict_txt}</span> "
        f"<span class='tag accent'>{int(d.efficiency*100)}% of ideal work-up</span> "
        f"<span class='tag miss'>{d.points} pts · turn {d.turns}</span></div></div>"
        f"<div class='feed-text' style='margin:10px 0 6px'>{html.escape(d.attending_note)} {llm_badge}</div>"
        + "".join(rows)
        + (f"<div class='dbrief-row'><span class='lbl'>Teaching</span></div>{teach}" if teach else "")
        + safety + cite
        + "</div>"
    )


def _trajectory_json(state) -> str:
    import json
    return json.dumps({
        "case_id": state.case_id,
        "specialty": state.specialty,
        "runtime": "langgraph" if _langgraph_active() else "deterministic-router",
        "spans": [e.to_otel_dict() for e in state.trace],
    }, indent=2)


# --------------------------------------------------------------------------- #
# Sidebar — case control
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("<div class='wordmark'>MedSim</div>"
                "<div class='subtitle'>Clinical Case Simulator</div><br>",
                unsafe_allow_html=True)

    st.markdown("**Specialties**")
    chosen = [s for s in config.SPECIALTIES if st.checkbox(s, value=(s == "Internal Medicine"))]
    gen_diff = st.selectbox("Difficulty", [1, 2, 3], index=1,
                            format_func=lambda x: {1: "1 — easier", 2: "2", 3: "3 — harder"}[x])

    if config.llm_available():
        if st.button("Start New Case", use_container_width=True):
            specialties = chosen or list(config.SPECIALTIES)
            spec = random.choice(specialties)
            with st.spinner(f"Authoring & verifying a new {spec} case…"):
                try:
                    ss.state = engine.create_generated_case(
                        spec, difficulty=gen_diff, store=ss.store)
                    st.rerun()
                except Exception as exc:  # CaseGenerationError / API errors
                    st.error(f"Generation failed: {exc}")
        provider = config.LLM_PROVIDER or ("gemini" if config.GOOGLE_API_KEY else "anthropic")
        st.caption(f"Every case is AI-authored & verified · provider: {provider} · "
                   f"{config.GEN_VERIFIERS} reviewers")
    else:
        st.button("Start New Case", use_container_width=True, disabled=True)
        st.caption("MedSim generates every case with AI. Set GOOGLE_API_KEY or "
                   "ANTHROPIC_API_KEY to begin.")

    st.divider()
    with st.expander("Agentic architecture"):
        st.markdown(
            "<div class='arch'>"
            "<b>6 specialised agents</b> routed by a <b>Supervisor</b> over a turn-based "
            "<b>LangGraph</b> state machine:"
            "<ul style='margin:6px 0 6px 16px;padding:0'>"
            "<li><b>Patient</b> — intake + history (grounded <code>LLM</code> dialogue)</li>"
            "<li><b>Clinical</b> — labs & drugs → <code>lab_simulator</code>, <code>drug_effect_engine</code></li>"
            "<li><b>Surgery</b> — <code>emergency_surgery</code> indication</li>"
            "<li><b>Knowledge</b> — RAG answers → <code>knowledge_lookup</code> (<code>LLM</code>)</li>"
            "<li><b>Critic</b> — grades the diagnosis</li>"
            "<li><b>Supervisor</b> — routes, scores, enforces retries</li>"
            "</ul>"
            "<b>LLM roles:</b> authors new cases (generate-then-verify), voices the "
            "patient, and synthesises knowledge — all grounded &amp; guarded.<br>"
            f"<b>Runtime:</b> {'LangGraph' if _langgraph_active() else 'built-in router'} · "
            "every action is a traced span."
            "</div>", unsafe_allow_html=True)
        st.code(engine.MERMAID, language="text")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.markdown("<div class='hero-title'>Hospital <span class='accent'>Simulation</span></div>"
            "<div class='subtitle'>Work the case with evidence-based decisions. "
            "Every point is graded against clinical guidelines.</div>", unsafe_allow_html=True)
st.write("")

state = ss.state
if state is None:
    if config.llm_available():
        st.markdown("<div class='card'>Every patient case is <b>authored and verified live by AI</b> "
                    "(constrained to a real drug/lab vocabulary, then checked by independent reviewer "
                    "agents). Pick specialties + difficulty in the sidebar and choose "
                    "<b>Start New Case</b> to generate one.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card'>MedSim <b>generates every case with AI</b> — there are no "
                    "hard-coded cases. Set <code>GOOGLE_API_KEY</code> or <code>ANTHROPIC_API_KEY</code> "
                    "(see <code>.env.example</code>) to begin.</div>", unsafe_allow_html=True)
    st.stop()

left, right = st.columns([1.15, 1], gap="large")

# ---- Patient card + scoreboard ------------------------------------------- #
with left:
    demo = state.demographics
    status_tag = {"active": "accent", "complete": "accent", "failed": "alert"}[state.status]
    _disease = data_loader.get_disease(state.disease_id) or {}
    gen_tag = ""
    if _disease.get("generated"):
        prov = _disease.get("provenance", {})
        conf = prov.get("verification", {}).get("confidence")
        conf_txt = f" · verify {conf}" if conf is not None else ""
        gen_tag = f"<span class='tag alert'>AI-generated · unreviewed{conf_txt}</span> "
        cv = prov.get("citation_verified")
        if cv is True:
            gen_tag += "<span class='tag ok' title='Guideline citation URL resolves'>citation verified</span> "
        elif cv is False:
            # Note: never render the model's raw URL here — its slug can spell out
            # the diagnosis and leak the answer mid-case.
            gen_tag += ("<span class='tag miss' title='The model-cited guideline URL did not "
                        "resolve and was replaced with a literature search'>citation unverified</span> ")
    sev = _disease.get("severity")
    sev_tag = ""
    if sev:
        sc = {1: "miss", 2: "accent", 3: "alert"}.get(int(sev), "miss")
        sl = {1: "low stakes", 2: "urgent", 3: "critical"}.get(int(sev), "")
        sev_tag = (f"<span class='tag {sc}' title='Case severity — scores are weighted "
                   f"by clinical stakes'>{sl}</span> ")
    st.markdown(
        f"<div class='card'>"
        f"{gen_tag}"
        f"<span class='tag {status_tag}'>{state.status}</span> "
        f"<span class='tag accent'>{html.escape(state.specialty)}</span> {sev_tag}<br><br>"
        f"<b>{html.escape(demo.get('age',''))} {html.escape(demo.get('sex',''))}</b>"
        f"<div class='chief'>Chief complaint: {html.escape(state.chief_complaint)}</div><br>"
        + "".join(
            f"<span class='vital'><div class='v'>{_fmt(v)}</div><div class='k'>{k}</div></span>"
            for k, v in state.vitals.items())
        + f"<br><br><span class='tag alert'>Allergies: "
          f"{html.escape(', '.join(state.allergies) or 'NKDA')}</span>"
        f"</div>", unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    s1.markdown(f"<div class='card center'><div class='metric'>{state.points}</div>"
                f"<div class='metric-label'>Case points</div></div>", unsafe_allow_html=True)
    s2.markdown(f"<div class='card center'><div class='metric'>{state.turn}</div>"
                f"<div class='metric-label'>Turn</div></div>", unsafe_allow_html=True)
    s3.markdown(f"<div class='card center'><div class='metric'>"
                f"{max(0, state.max_retries - state.retries)}</div>"
                f"<div class='metric-label'>Diagnosis tries left</div></div>",
                unsafe_allow_html=True)

# ---- Action panels -------------------------------------------------------- #
with right:
    disabled = state.status != "active"
    tab_hx, tab_dx, tab_rx, tab_kb = st.tabs(
        ["History & Exam", "Tests", "Medications & Surgery", "Knowledge & Diagnosis"])

    with tab_hx:
        with st.form("hx_form", clear_on_submit=True):
            q = st.text_input("Ask a history question",
                              placeholder="e.g. When did the pain start?")
            if st.form_submit_button("Ask", disabled=disabled) and q:
                _do({"type": "ask_history", "payload": q})
                st.rerun()
        exams = st.multiselect("Physical examination", config.PHYSICAL_EXAMS)
        if st.button("Perform examination", disabled=disabled or not exams):
            for e in exams:
                _do({"type": "perform_exam", "payload": e})
            st.rerun()

    with tab_dx:
        tests = st.multiselect("Order tests / imaging", engine.available_tests())
        if st.button("Order selected tests", disabled=disabled or not tests):
            for t in tests:
                _do({"type": "order_test", "payload": t})
            st.rerun()

    with tab_rx:
        drug = st.selectbox("Prescribe medication", [""] + engine.available_drugs())
        if st.button("Prescribe", disabled=disabled or not drug):
            _do({"type": "prescribe_drug", "payload": drug})
            st.rerun()
        proc = st.selectbox("Emergency surgery", [""] + engine.available_procedures())
        if st.button("Request surgery", disabled=disabled or not proc):
            _do({"type": "request_surgery", "payload": proc})
            st.rerun()

    with tab_kb:
        with st.form("kb_form", clear_on_submit=True):
            kq = st.text_input("Ask a knowledge question",
                               placeholder="e.g. first-line treatment for pyelonephritis?")
            if st.form_submit_button("Look up", disabled=disabled) and kq:
                _do({"type": "knowledge_query", "payload": kq})
                st.rerun()
        with st.form("dx_form", clear_on_submit=True):
            dx = st.text_input("Final diagnosis (your commitment)")
            ddx = st.text_input("Ranked differentials — optional, comma-separated",
                                help="Real diagnosis is probabilistic. List your top considerations, "
                                     "most likely first — you earn recognition credit if the true "
                                     "diagnosis is among them.")
            conf = st.slider("Confidence", 0, 100, 50,
                             help="Calibration is scored: reward for being right and confident, "
                                  "and for hedging when wrong; penalty for confident errors.")
            if st.form_submit_button("Submit diagnosis", disabled=disabled) and dx:
                diffs = [d.strip() for d in ddx.split(",") if d.strip()][:3]
                _do({"type": "submit_diagnosis", "payload": dx,
                     "differentials": diffs, "confidence": conf})
                st.rerun()

# ---- End-of-case debrief -------------------------------------------------- #
if state.debrief is not None:
    st.markdown(_debrief_html(state.debrief), unsafe_allow_html=True)

# ---- Feed ----------------------------------------------------------------- #
st.markdown("<div class='section-head'>Case log</div>", unsafe_allow_html=True)
for m in reversed(state.feed):
    delta = ""
    if m.points_delta:
        cls = "delta-gain" if m.points_delta > 0 else "delta-loss"
        delta = f" · <span class='{cls}'>{'+' if m.points_delta > 0 else ''}{m.points_delta}</span>"
    cites = ""
    if m.citations:
        multi = len(m.citations) > 1
        cites = "<div class='cite'>" + "".join(
            f"<a href='{html.escape(c)}' target='_blank'>reference{(' ' + str(i + 1)) if multi else ''}</a>"
            for i, c in enumerate(m.citations)) + "</div>"
    llm_badge = "<span class='badge llm'>LLM</span>" if getattr(m, "llm", False) else ""
    st.markdown(
        f"<div class='feed-item {m.kind}'>"
        f"<span class='feed-role'>{html.escape(m.role)} · turn {m.turn}</span>{llm_badge}{delta}"
        f"<div class='feed-text'>{html.escape(m.text)}</div>{cites}</div>",
        unsafe_allow_html=True)

# ---- Agent trajectory (observability) ------------------------------------ #
if state.trace:
    import base64
    from src import trace as _trace
    from src import viz as _viz
    s = _trace.summary(state.trace)
    with st.expander(f"Agent trajectory · {s['actions']} steps · "
                     f"{len(s['agents'])} agents · {s['llm_actions']} LLM calls", expanded=False):
        st.caption("Multi-agent routing — nodes light up along the path each action took:")
        _b64 = base64.b64encode(_viz.routing_graph_svg(state.trace).encode("utf-8")).decode("ascii")
        st.markdown(f"<img alt='agent routing graph' style='width:100%;max-width:660px' "
                    f"src='data:image/svg+xml;base64,{_b64}'/>", unsafe_allow_html=True)
        rows = "".join(
            f"<tr><td class='mono'>{e.turn}</td><td>{html.escape(e.router)} → "
            f"<b>{html.escape(e.agent or '—')}</b></td>"
            f"<td class='mono'>{html.escape(', '.join(e.tools) or '—')}</td>"
            f"<td>{html.escape(e.action)}"
            f"{' <span class=\"badge llm\">LLM</span>' if e.llm_used else ''}</td>"
            f"<td class='mono'>{'+' if e.points_delta >= 0 else ''}{e.points_delta}</td>"
            f"<td class='mono'>{e.duration_ms:.0f} ms</td></tr>"
            for e in state.trace)
        st.markdown(
            "<table class='traj'><tr><th>Turn</th><th>Route</th><th>Tools</th>"
            "<th>Action</th><th>Δpts</th><th>Latency</th></tr>"
            f"{rows}</table>", unsafe_allow_html=True)
        st.caption("Every action is a traced span (OpenTelemetry-shaped). "
                   "Runtime router: " + ("LangGraph" if _langgraph_active() else "built-in"))
        st.download_button("Download trajectory (JSON)",
                           data=_trajectory_json(state),
                           file_name=f"trajectory_{state.case_id}.json",
                           mime="application/json")
