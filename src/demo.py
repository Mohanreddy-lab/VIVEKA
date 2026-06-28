"""
demo.py — VIVEKA Streamlit Demo

Features:
  • Parallel LLM reranking — all candidates scored simultaneously (5–10× faster)
  • Live streaming — watch rankings update as results arrive
  • Interview question generator — tailored questions per candidate
  • Ideal candidate blueprint — AI describes the perfect hire
  • Outreach email drafts — one-click personalised LinkedIn messages
  • Radar charts — signal breakdown per candidate
  • 4 analytics tabs: Score Breakdown · Score Trend · Skill Gap · Compare
  • Download ranked CSV with one click

Run with:  streamlit run src/demo.py
"""

import csv
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from agent  import load_profiles
from output import normalize_scores

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VIVEKA — Intelligent Candidate Ranking",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .metric-card { background: #1e2a3a; border-radius: 8px; padding: 12px; }
  .gem-badge   { color: #ffd700; font-weight: bold; }
  .score-high  { color: #4caf50; }
  .score-med   { color: #ff9800; }
  .score-low   { color: #f44336; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 VIVEKA — Discerning True Talent from Noise")
st.caption("Offline · Free · Private · Powered by Ollama locally")

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    model_name = st.text_input(
        "Ollama model",
        value=os.getenv("MANTHAN_MODEL", "llama3.2"),
        help="Must be pulled: `ollama pull llama3.2`",
    )
    os.environ["MANTHAN_MODEL"] = model_name

    rerank_n = st.slider(
        "Candidates to rerank (LLM calls)",
        min_value=3, max_value=50, value=8,
        help="Lower = faster. Raise to 50 for final runs.",
    )

    parallel_mode = st.toggle(
        "Parallel reranking (faster)",
        value=False,
        help="Score all candidates simultaneously. Faster on multi-core machines but may cause memory pressure with large models.",
    )

    max_workers = st.slider(
        "Parallel workers",
        min_value=2, max_value=10, value=5,
        disabled=not parallel_mode,
        help="Simultaneous LLM calls. 5 is sweet spot for most hardware.",
    )
    os.environ["MANTHAN_PARALLEL_WORKERS"] = str(max_workers)

    st.divider()
    st.subheader("Signal Weights")
    st.caption("Auto-normalised. Drag to adjust contribution to composite score.")

    w_embed     = st.slider("Embedding (semantic)",  0.0, 1.0, 0.30, 0.05)
    w_skill     = st.slider("Skill match",           0.0, 1.0, 0.40, 0.05)
    w_seniority = st.slider("Seniority fit",         0.0, 1.0, 0.15, 0.05)
    w_activity  = st.slider("Activity / behavior",   0.0, 1.0, 0.15, 0.05)

    os.environ["MANTHAN_W_EMBED"]     = str(w_embed)
    os.environ["MANTHAN_W_SKILL"]     = str(w_skill)
    os.environ["MANTHAN_W_SENIORITY"] = str(w_seniority)
    os.environ["MANTHAN_W_ACTIVITY"]  = str(w_activity)

    st.divider()
    st.subheader("📁 Upload Profiles")
    uploaded = st.file_uploader(
        "Upload profiles.json or profiles.csv",
        type=["json", "csv"],
        help="Drag & drop your candidate profiles file. Replaces the built-in demo data.",
    )
    if uploaded is not None:
        data_dir.mkdir(exist_ok=True)
        suffix = ".json" if uploaded.name.endswith(".json") else ".csv"
        dest = data_dir / f"profiles{suffix}"
        dest.write_bytes(uploaded.read())
        kb = dest.stat().st_size // 1024 + 1
        st.success(f"Saved {uploaded.name} ({kb} KB) — cache will auto-refresh.")
        st.rerun()

    st.divider()
    st.subheader("🔒 Trust & Safety")

    pii_on = st.toggle(
        "PII Firewall (strip identity fields)",
        value=True,
        help="Removes name, gender, age, location before scoring. Score the work, not the identity.",
    )
    os.environ["MANTHAN_PII_FIREWALL"] = "on" if pii_on else "off"
    if pii_on:
        st.success("PII firewall ON — identity fields stripped before scoring.")
    else:
        st.warning("PII firewall OFF — identity fields visible to model.")

    audit_on = st.toggle(
        "Audit trail (log decisions to audit.jsonl)",
        value=True,
        help="Writes every ranking decision to data/audit.jsonl for reproducibility.",
    )
    os.environ["MANTHAN_AUDIT"] = "on" if audit_on else "off"

    st.divider()
    if st.button("🔍 Check Ollama connection"):
        from llm import check_ollama
        ok, msg = check_ollama()
        if ok:
            st.success(msg)
        else:
            st.error(msg)
            st.code("ollama serve", language="bash")

    st.divider()
    st.markdown("**Switch model:**  `ollama pull mistral:7b`")
    st.markdown("**Fast model:**    `ollama pull llama3.2`")
    st.markdown("**Stop server:**   `Ctrl+C` in terminal")

# ---------------------------------------------------------------------------
# Candidate data source
# ---------------------------------------------------------------------------

data_dir  = Path(__file__).parent.parent / "data"
json_path = data_dir / "profiles.json"
csv_path  = data_dir / "profiles.csv"

if json_path.exists():
    n = len(json.loads(json_path.read_text(encoding="utf-8")))
    data_source = f"📄 profiles.json  ({n} candidates)"
elif csv_path.exists():
    data_source = "📄 profiles.csv"
else:
    data_source = "🔧 Built-in demo profiles  (8 candidates)"

st.caption(f"Candidate data: {data_source}")

# ---------------------------------------------------------------------------
# Job description input
# ---------------------------------------------------------------------------

st.subheader("Job Description")

_SAMPLE_JD = """Senior Data Engineer — Platform Team

We are looking for a Senior Data Engineer to join our data platform team.

Requirements:
- 5+ years data engineering experience
- Strong Python and SQL
- Apache Spark and Airflow
- AWS (S3, Glue, Redshift)
- dbt experience

Nice to have: Kafka, MLflow, feature stores.

Responsibilities: own pipelines end-to-end, mentor junior engineers,
work with ML and product teams. Ambiguity is normal here."""

jd_text = st.text_area(
    "Paste the job description here",
    value=_SAMPLE_JD,
    height=220,
)

# ---------------------------------------------------------------------------
# FAISS index cache — keyed on file hash
# ---------------------------------------------------------------------------

def _profile_hash(data_dir: Path) -> str:
    for fname in ("profiles.json", "profiles.csv"):
        p = data_dir / fname
        if p.exists():
            return hashlib.md5(p.read_bytes()).hexdigest()
    return "demo"


@st.cache_resource(show_spinner="Loading embedding model and indexing profiles…")
def _get_index(data_dir_key: str, _file_hash: str):
    from recall import RecallEngine
    from agent  import load_profiles as _lp
    _profiles = _lp(Path(data_dir_key))
    eng = RecallEngine()
    eng.index_candidates(_profiles)
    return eng, _profiles


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _conf_badge(conf: str) -> str:
    return {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(str(conf).lower(), "⚪")


def _score_color(score_100: float) -> str:
    if score_100 >= 70:
        return "score-high"
    if score_100 >= 45:
        return "score-med"
    return "score-low"


def _render_candidate_card(i: int, c: dict, expanded: bool = False, render_id: int = 0) -> None:
    cid    = c.get("candidate_id") or c.get("id") or f"#{i}"
    title  = c.get("title", "Unknown")
    s100   = c.get("score_100", 0.0)
    llm    = c.get("llm_score",  "…")
    conf   = str(c.get("confidence", "…"))
    cal_c  = c.get("calibrated_confidence")
    reason = c.get("reason", "Scoring in progress…")
    is_gem = c.get("hidden_gem", False)
    ev     = c.get("skill_evidence", {})
    stuffing = c.get("stuffing", {})
    cf       = c.get("counterfactual", {})

    gem_badge = "  ★ Hidden Gem" if is_gem else ""
    label = f"**#{i}  {cid}**  |  {title}{gem_badge}  |  Score: **{s100:.1f}/100**"

    with st.expander(label, expanded=expanded):
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Score /100",  f"{s100:.1f}")
        col2.metric("LLM Score",   f"{llm}/10" if isinstance(llm, int) else str(llm))
        col3.metric("Confidence",  f"{_conf_badge(conf)} {conf.title()}")
        col4.metric("Skill Match", f"{c.get('skill_score', 0):.0%}")
        col5.metric("Seniority",   f"{c.get('seniority_score', 0):.0%}")

        # Calibrated confidence bar
        if cal_c is not None:
            cal_pct = int(cal_c * 100)
            bar_color = "#4caf50" if cal_pct >= 70 else ("#ff9800" if cal_pct >= 45 else "#f44336")
            st.markdown(
                f"**Calibrated Confidence:** "
                f"<span style='color:{bar_color};font-weight:bold'>{cal_pct}%</span>  "
                f"<progress value='{cal_pct}' max='100' style='width:160px'></progress>",
                unsafe_allow_html=True,
            )

        st.markdown(f"**Reason:** {reason}")

        # Citation evidence
        ev_verified    = c.get("evidence_verified",    [])
        ev_unsupported = c.get("evidence_unsupported", [])
        if ev_verified or ev_unsupported:
            with st.expander("📎 Citation Evidence"):
                if ev_verified:
                    for snippet in ev_verified:
                        st.success(f'✅ **Verified:** "{snippet}"')
                if ev_unsupported:
                    for snippet in ev_unsupported:
                        st.error(f'⚠️ **Not found in profile:** "{snippet}"')

        if c.get("analysis"):
            with st.expander("🧠 Chain-of-thought reasoning"):
                st.markdown(c["analysis"])

        ev_hit  = ev.get("required_matched", [])
        ev_miss = ev.get("required_missing", [])
        if ev_hit or ev_miss:
            h_col, m_col = st.columns(2)
            h_col.markdown("✅ **Skills matched:** " + (", ".join(ev_hit) or "none"))
            m_col.markdown("❌ **Skills missing:** " + (", ".join(ev_miss) or "none"))

        # Keyword-stuffing alert
        if stuffing and stuffing.get("stuffing_ratio", 0) > 0.2:
            unsup = stuffing.get("claimed_unsupported", [])
            ratio = stuffing.get("stuffing_ratio", 0)
            st.warning(
                f"⚠️ **Résumé stuffing detected** ({ratio:.0%} of listed skills have no narrative support): "
                + ", ".join(unsup[:5])
            )

        # Counterfactual / ceiling
        if cf and cf.get("ceiling"):
            st.info(f"🔮 **Why not higher?** {cf['ceiling']}")

        # Radar chart
        try:
            import plotly.graph_objects as go
            r_vals = [
                c.get("embedding_score",  0),
                c.get("skill_score",      0),
                c.get("seniority_score",  0),
                c.get("activity_score",   0),
                (c.get("llm_score", 0) or 0) / 10.0,
                c.get("calibrated_confidence", 0),
            ]
            theta = ["Semantic", "Skills", "Seniority", "Activity", "LLM Fit", "Cal.Conf"]
            fig = go.Figure(go.Scatterpolar(
                r=r_vals + [r_vals[0]],
                theta=theta + [theta[0]],
                fill="toself",
                fillcolor="rgba(78,154,241,0.2)",
                line=dict(color="#4e9af1"),
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
                showlegend=False,
                height=260,
                margin=dict(l=30, r=30, t=30, b=30),
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True, key=f"radar_{render_id}_{i}_{cid[:8]}")
        except ImportError:
            pass


def _render_shortlist(ranked: list, top_n: int = 20, render_id: int = 0) -> None:
    gems = [c for c in ranked if c.get("hidden_gem")]
    if gems:
        st.success(f"★ {len(gems)} Hidden Gem(s) surfaced so far")
    for i, c in enumerate(ranked[:top_n], 1):
        _render_candidate_card(i, c, expanded=(i <= 3), render_id=render_id)


# ---------------------------------------------------------------------------
# Analytics charts
# ---------------------------------------------------------------------------

def _show_score_breakdown(ranked: list) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("Install plotly: `pip install plotly`")
        return

    top  = ranked[:10]
    ids  = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(top)]

    signals = {
        "Embedding":  [c.get("embedding_score",  0) for c in top],
        "Skill":      [c.get("skill_score",       0) for c in top],
        "Seniority":  [c.get("seniority_score",   0) for c in top],
        "Activity":   [c.get("activity_score",    0) for c in top],
    }
    colors = {"Embedding": "#4e9af1", "Skill": "#7ce38b", "Seniority": "#f0ad4e", "Activity": "#e07db3"}

    fig = go.Figure()
    for sig, vals in signals.items():
        fig.add_trace(go.Bar(name=sig, x=ids, y=vals, marker_color=colors[sig]))

    composites = [c.get("composite_score", 0) for c in top]
    fig.add_trace(go.Scatter(
        x=ids, y=composites, name="Composite",
        mode="lines+markers", line=dict(color="#ffffff", width=2, dash="dot"),
    ))
    fig.update_layout(
        barmode="stack",
        title="Signal Contribution (stacked) + Composite (line)",
        yaxis=dict(range=[0, 1.2], title="Score"),
        xaxis_tickangle=-30,
        template="plotly_dark",
        height=400,
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_score_trend(ranked: list) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    ids    = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(ranked)]
    finals = [c.get("final_score", 0) for c in ranked]
    comps  = [c.get("composite_score", 0) for c in ranked]
    x      = list(range(1, len(ranked) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=finals, name="Final Score", mode="lines+markers",
        line=dict(color="#4e9af1", width=2),
        text=ids, hovertemplate="%{text}: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=comps, name="Composite", mode="lines",
        line=dict(color="#7ce38b", width=1, dash="dot"),
    ))
    fig.add_hrect(y0=0.6, y1=1.0, fillcolor="#4e9af1", opacity=0.06, annotation_text="Strong zone")
    fig.update_layout(
        title="Score Dropoff by Rank", xaxis_title="Rank",
        yaxis=dict(range=[0, 1.05], title="Score"),
        template="plotly_dark", height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_skill_gap(ranked: list, parsed_jd: dict) -> None:
    import pandas as pd

    top5        = ranked[:5]
    req_skills  = parsed_jd.get("required_skills", [])
    impl_skills = parsed_jd.get("implied_skills",  [])
    all_skills  = req_skills + impl_skills

    if not all_skills or not top5:
        st.info("Not enough data for skill gap matrix.")
        return

    rows = []
    for skill in all_skills:
        row  = {"Skill": skill, "Type": "required" if skill in req_skills else "implied"}
        for c in top5:
            cid = c.get("candidate_id") or c.get("id") or "?"
            ev  = c.get("skill_evidence", {})
            matched = [s.lower() for s in ev.get("required_matched", [])]
            row[cid[:10]] = "✅" if skill.lower() in matched else "❌"
        rows.append(row)

    df = pd.DataFrame(rows)
    style_fn = lambda v: "background-color: #1a4a1a" if v == "✅" else "background-color: #4a1a1a"
    value_cols = [c for c in df.columns if c not in ("Skill", "Type")]
    try:
        styled = df.style.map(style_fn, subset=value_cols)
    except AttributeError:
        styled = df.style.applymap(style_fn, subset=value_cols)
    st.dataframe(
        styled,
        use_container_width=True,
        height=min(40 * len(rows) + 80, 600),
    )


def _show_stuffing_analysis(ranked: list) -> None:
    """Table of stuffing_ratio per candidate with unsupported skill details."""
    import pandas as pd

    rows = []
    for i, c in enumerate(ranked[:15], 1):
        cid      = c.get("candidate_id") or c.get("id") or f"#{i}"
        stuffing = c.get("stuffing", {})
        ratio    = stuffing.get("stuffing_ratio", 0.0)
        unsup    = ", ".join(stuffing.get("claimed_unsupported", [])[:5]) or "—"
        rows.append({
            "Rank": i,
            "Candidate": cid,
            "Title": c.get("title", "")[:25],
            "Stuffing %": f"{ratio:.0%}",
            "Unsupported Claims": unsup,
        })

    if not rows:
        st.info("No candidates to show.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    try:
        import plotly.graph_objects as go
        ids    = [r["Candidate"] for r in rows]
        ratios = [c.get("stuffing", {}).get("stuffing_ratio", 0.0) for c in ranked[:15]]
        colors = ["#f44336" if r > 0.5 else ("#ff9800" if r > 0.2 else "#4caf50") for r in ratios]

        fig = go.Figure(go.Bar(
            x=ids, y=[r * 100 for r in ratios],
            marker_color=colors,
            text=[f"{r:.0%}" for r in ratios],
            textposition="outside",
        ))
        fig.update_layout(
            title="Résumé Stuffing Ratio (% of listed skills with no narrative support)",
            yaxis=dict(title="Stuffing %", range=[0, 110]),
            xaxis_tickangle=-30,
            template="plotly_dark",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🟢 < 20% — genuine  |  🟡 20–50% — borderline  |  🔴 > 50% — likely padded")
    except ImportError:
        pass


def _show_citation_audit(ranked: list) -> None:
    """Show per-candidate citation verification results."""
    import pandas as pd

    rows = []
    for i, c in enumerate(ranked[:15], 1):
        cid      = c.get("candidate_id") or c.get("id") or f"#{i}"
        verified = c.get("evidence_verified",    [])
        unsup    = c.get("evidence_unsupported", [])
        cal_c    = c.get("calibrated_confidence", 0.0)
        rows.append({
            "Rank": i,
            "Candidate": cid[:12],
            "Verified Snippets": len(verified),
            "Hallucinated Snippets": len(unsup),
            "Cal. Confidence": f"{cal_c:.0%}",
            "Sample Verified": (verified[0][:50] if verified else "—"),
            "Sample Hallucinated": (unsup[0][:50] if unsup else "—"),
        })

    if not rows:
        st.info("No citation data yet — run the pipeline first.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    n_hallucinated = sum(1 for r in rows if r["Hallucinated Snippets"] > 0)
    n_verified     = sum(r["Verified Snippets"] for r in rows)
    n_unsup        = sum(r["Hallucinated Snippets"] for r in rows)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total verified citations",     n_verified)
    m2.metric("Total hallucinated citations", n_unsup,    delta=f"-{n_hallucinated} candidates affected" if n_hallucinated else None)
    m3.metric("Candidates with hallucinations", n_hallucinated)

    if n_hallucinated:
        st.warning(
            f"⚠️ {n_hallucinated} candidate(s) had citations not found in their profile. "
            "Their confidence was automatically downgraded."
        )
    else:
        st.success("✅ All citations verified — no hallucinated evidence detected.")


def _show_comparison(ranked: list) -> None:
    if len(ranked) < 2:
        st.info("Need at least 2 ranked candidates to compare.")
        return

    ids    = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(ranked)]
    id_to_c = {i: c for i, c in zip(ids, ranked)}

    col1, col2 = st.columns(2)
    with col1:
        a_id = st.selectbox("Candidate A", ids, index=0, key="cmp_a")
    with col2:
        b_id = st.selectbox("Candidate B", ids, index=min(1, len(ids)-1), key="cmp_b")

    if a_id == b_id:
        st.warning("Select two different candidates.")
        return

    a, b = id_to_c[a_id], id_to_c[b_id]

    try:
        import plotly.graph_objects as go
        signals = ["Embedding", "Skill", "Seniority", "Activity", "LLM (/10)"]
        a_vals  = [
            a.get("embedding_score",  0), a.get("skill_score",  0),
            a.get("seniority_score",  0), a.get("activity_score", 0),
            (a.get("llm_score", 0) or 0) / 10.0,
        ]
        b_vals  = [
            b.get("embedding_score",  0), b.get("skill_score",  0),
            b.get("seniority_score",  0), b.get("activity_score", 0),
            (b.get("llm_score", 0) or 0) / 10.0,
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(name=a_id[:12], x=signals, y=a_vals, marker_color="#4e9af1"))
        fig.add_trace(go.Bar(name=b_id[:12], x=signals, y=b_vals, marker_color="#f0ad4e"))
        fig.update_layout(
            title=f"{a_id[:12]} vs {b_id[:12]}",
            barmode="group", yaxis=dict(range=[0, 1.05]),
            template="plotly_dark", height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        pass

    ca, cb = st.columns(2)
    for col, c, cid in [(ca, a, a_id), (cb, b, b_id)]:
        with col:
            st.markdown(f"**{cid}** — {c.get('title', '')}")
            st.metric("Score /100",  f"{c.get('score_100', 0):.1f}")
            st.metric("LLM Score",   f"{c.get('llm_score', '—')}/10")
            st.metric("Confidence",  str(c.get('confidence', '—')).title())
            ev = c.get("skill_evidence", {})
            st.markdown("✅ " + ", ".join(ev.get("required_matched", []) or ["none"]))
            st.markdown("❌ " + ", ".join(ev.get("required_missing", []) or ["none"]))
            st.markdown(f"*{c.get('reason', '')}*")


# ---------------------------------------------------------------------------
# Ideal candidate renderer
# ---------------------------------------------------------------------------

def _show_ideal_candidate(ideal: dict) -> None:
    st.markdown(f"**Profile:** {ideal.get('summary', '')}")

    col1, col2, col3 = st.columns(3)
    scarcity = ideal.get("estimated_market_scarcity", "unknown")
    scarcity_color = {"rare": "🔴", "uncommon": "🟡", "available": "🟢"}.get(scarcity, "⚪")
    col1.metric("Market Scarcity", f"{scarcity_color} {scarcity.title()}")

    must_have = ideal.get("must_have", [])
    if must_have:
        col2.markdown("**Must-haves:**\n" + "\n".join(f"• {s}" for s in must_have))

    differentiators = ideal.get("differentiators", [])
    if differentiators:
        col3.markdown("**Differentiators:**\n" + "\n".join(f"• {s}" for s in differentiators))

    c1, c2 = st.columns(2)
    c1.info(f"**Hidden gem signal:** {ideal.get('hidden_gem_signal', '—')}")
    c2.warning(f"**Red flag to watch:** {ideal.get('red_flag', '—')}")


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _make_csv(ranked: list) -> str:
    buf    = io.StringIO()
    fields = ["rank", "candidate_id", "title", "score_100", "final_score",
              "llm_score", "confidence", "hidden_gem", "reason",
              "skill_score", "seniority_score", "activity_score", "composite_score"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for i, c in enumerate(ranked, 1):
        row = dict(c)
        row["rank"]         = i
        row["candidate_id"] = c.get("candidate_id") or c.get("id", "")
        writer.writerow(row)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------

run_btn = st.button("▶  Run VIVEKA", type="primary", use_container_width=True)

if run_btn:
    if len(jd_text.strip()) < 30:
        st.error("Job description is too short. Please paste the full text (at least 30 characters).")
        st.stop()

    # ── Ollama health check ────────────────────────────────────────────────────
    from llm import check_ollama
    with st.spinner("Checking Ollama…"):
        _ollama_ok, _ollama_msg = check_ollama()
    if not _ollama_ok:
        st.error(f"Cannot reach Ollama: {_ollama_msg}")
        st.info("Fix: open a new terminal and run `ollama serve`, then click Run again.")
        st.code("ollama serve", language="bash")
        st.stop()

    # ── Stage 1–3 ─────────────────────────────────────────────────────────────
    t0 = time.time()
    with st.status("Running Stages 1–3…", expanded=True) as status:
        st.write("Stage 1 — Parsing job description with local LLM…")
        try:
            from jd_parser import parse_jd
            parsed_jd = parse_jd(jd_text)
        except Exception as e:
            st.error(f"Stage 1 (JD parsing) failed: {type(e).__name__}: {e}")
            st.info("Make sure Ollama is running and the model is pulled: `ollama pull llama3.2`")
            st.stop()

        st.write("Stage 2 — FAISS recall (index cached; auto-invalidates on file change)…")
        try:
            engine, profiles = _get_index(str(data_dir), _profile_hash(data_dir))
            recalled = engine.recall(parsed_jd, top_k=min(200, len(profiles)))
        except Exception as e:
            st.error(f"Stage 2 (embedding/recall) failed: {type(e).__name__}: {e}")
            st.stop()

        st.write("Stage 3 — Multi-signal scoring (4 signals, synonym-aware)…")
        try:
            from scoring import score_candidates
            scored = score_candidates(recalled, parsed_jd, top_n=50)
        except Exception as e:
            st.error(f"Stage 3 (scoring) failed: {type(e).__name__}: {e}")
            st.stop()
        mode_label = "parallel" if parallel_mode else "sequential"
        status.update(
            label=f"Stages 1–3 done  →  starting LLM rerank ({mode_label}, {len(scored)} candidates)",
            state="running",
        )

    # ── JD summary ────────────────────────────────────────────────────────────
    st.subheader("📋 Parsed Job Requirements")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Seniority",        parsed_jd.get("seniority", "—").title())
    m2.metric("Required skills",  len(parsed_jd.get("required_skills", [])))
    m3.metric("Implied skills",   len(parsed_jd.get("implied_skills",  [])))
    m4.metric("Candidates scored", len(scored))
    with st.expander("Full parsed JD"):
        st.json(parsed_jd)

    # ── Stage 4: rerank ───────────────────────────────────────────────────────
    from rerank import rerank_stream_parallel, rerank_stream

    st.subheader("🏆 Live Rankings — updating as model scores each candidate")
    progress   = st.progress(0, text="Starting LLM rerank…")
    eta_ph     = st.empty()
    results_ph = st.empty()

    ranked = []
    total  = min(rerank_n, len(scored))
    t_rerank = time.time()

    stream_fn = rerank_stream_parallel if parallel_mode else rerank_stream

    for i, result in enumerate(stream_fn(scored, parsed_jd, top_n=rerank_n), 1):
        ranked.append(result)
        ranked_sorted = sorted(ranked, key=lambda x: x["final_score"], reverse=True)
        ranked_sorted = normalize_scores(ranked_sorted)

        elapsed = time.time() - t_rerank
        rate    = i / elapsed if elapsed > 0 else 0
        remaining = (total - i) / rate if rate > 0 else 0
        eta_text  = f"⚡ {i}/{total} done · {elapsed:.0f}s elapsed · ~{remaining:.0f}s remaining"

        progress.progress(i / total, text=f"Scored {i}/{total} candidates…")
        eta_ph.caption(eta_text)

        with results_ph.container():
            _render_shortlist(ranked_sorted, top_n=20, render_id=i)

    t_total = time.time() - t0
    progress.progress(1.0, text=f"✅ Done in {t_total:.0f}s  —  {len(ranked_sorted)} candidates ranked")
    eta_ph.empty()

    # ── Analytics ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Analytics")

    tab_breakdown, tab_trend, tab_gap, tab_compare, tab_stuffing, tab_cite = st.tabs([
        "Score Breakdown", "Score Trend", "Skill Gap", "Compare",
        "Stuffing Detector", "Citation Audit",
    ])
    with tab_breakdown:
        st.caption("How each signal (embedding, skill, seniority, activity) contributed for the top 10.")
        _show_score_breakdown(ranked_sorted)
    with tab_trend:
        st.caption("How scores fall off across all ranked candidates — shows how clear-cut the top is.")
        _show_score_trend(ranked_sorted)
    with tab_gap:
        st.caption("Required and implied skills vs top 5 candidates — green = match, red = missing.")
        _show_skill_gap(ranked_sorted, parsed_jd)
    with tab_compare:
        st.caption("Pick any two candidates to compare scores, skills, and reasons side by side.")
        _show_comparison(ranked_sorted)
    with tab_stuffing:
        st.caption("Résumé-padding detection: skills listed but never mentioned in narrative prose.")
        _show_stuffing_analysis(ranked_sorted)
    with tab_cite:
        st.caption("Citation audit: which LLM evidence snippets were verified vs hallucinated.")
        _show_citation_audit(ranked_sorted)

    # ── Ideal Candidate Blueprint ──────────────────────────────────────────────
    st.divider()
    st.subheader("🎯 Ideal Candidate Blueprint")
    st.caption("AI-generated profile of the perfect hire for this role.")

    if st.button("Generate Ideal Candidate Profile", key="ideal_btn"):
        from rerank import generate_ideal_candidate
        with st.spinner("Generating ideal candidate blueprint…"):
            ideal = generate_ideal_candidate(parsed_jd)
        _show_ideal_candidate(ideal)

    # ── Interview Questions ───────────────────────────────────────────────────
    st.divider()
    st.subheader("🎤 Interview Question Generator")
    st.caption("Tailored questions for each candidate based on their specific profile and gaps.")

    if ranked_sorted:
        ids = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(ranked_sorted)]
        selected_id = st.selectbox("Select candidate", ids, key="iq_select")
        selected_c  = ranked_sorted[ids.index(selected_id)]

        if st.button("Generate Interview Questions", key="iq_btn"):
            from rerank import generate_interview_questions
            with st.spinner(f"Generating questions for {selected_id}…"):
                questions = generate_interview_questions(selected_c, parsed_jd)

            st.success(f"5 tailored questions for **{selected_id}** ({selected_c.get('title', '')})")
            for i, q in enumerate(questions, 1):
                q_type = "🔧 Technical" if i <= 2 else ("💬 Behavioral" if i <= 4 else "🎯 Fit")
                st.markdown(f"**{q_type} Q{i}:** {q}")

    # ── Outreach Email ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("✉️ Outreach Email Drafts")
    st.caption("Personalised LinkedIn message for a shortlisted candidate.")

    if ranked_sorted:
        out_ids = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(ranked_sorted[:5])]
        out_sel = st.selectbox("Select top candidate", out_ids, key="out_select")
        out_c   = ranked_sorted[out_ids.index(out_sel)]

        if st.button("Draft Outreach Message", key="out_btn"):
            from rerank import generate_outreach_message
            with st.spinner(f"Drafting message for {out_sel}…"):
                message = generate_outreach_message(out_c, parsed_jd)
            st.text_area("Copy and paste:", value=message, height=120, key="out_text")

    # ── Download ───────────────────────────────────────────────────────────────
    st.divider()
    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button(
        "⬇  Download ranked_output.csv",
        data=_make_csv(ranked_sorted),
        file_name="ranked_output.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col_dl2.download_button(
        "⬇  Download full JSON",
        data=json.dumps(ranked_sorted, indent=2, default=str),
        file_name="ranked_output.json",
        mime="application/json",
        use_container_width=True,
    )
