"""
demo.py — Streamlit Demo UI

Features:
  • Live-streaming rerank — see candidates appear as the model scores them
  • Score breakdown charts — grouped bar showing all 4 signals per candidate
  • Skill gap matrix — required skills vs top candidates, hit / miss
  • Candidate comparison — pick any two and compare side-by-side
  • Download CSV with one click

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
    page_title="MANTHAN — Candidate Ranking",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 MANTHAN — Intelligent Candidate Ranking")
st.caption("Offline · Free · Private · Powered by Ollama + Llama locally")

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    model_name = st.text_input(
        "Ollama model",
        value=os.getenv("MANTHAN_MODEL", "llama3.2"),
        help="Must be pulled with `ollama pull <model>`",
    )
    os.environ["MANTHAN_MODEL"] = model_name

    rerank_n = st.slider(
        "Candidates to rerank (LLM calls)",
        min_value=3, max_value=50, value=10,
        help="Lower = faster. Raise to 50 for the final run.",
    )

    st.divider()
    st.subheader("Signal Weights")
    st.caption("Drag to change how each signal contributes to the composite score. They are auto-normalised.")

    w_embed     = st.slider("Embedding (semantic)",  0.0, 1.0, 0.30, 0.05)
    w_skill     = st.slider("Skill match",           0.0, 1.0, 0.40, 0.05)
    w_seniority = st.slider("Seniority fit",         0.0, 1.0, 0.15, 0.05)
    w_activity  = st.slider("Activity / behavior",   0.0, 1.0, 0.15, 0.05)

    os.environ["MANTHAN_W_EMBED"]     = str(w_embed)
    os.environ["MANTHAN_W_SKILL"]     = str(w_skill)
    os.environ["MANTHAN_W_SENIORITY"] = str(w_seniority)
    os.environ["MANTHAN_W_ACTIVITY"]  = str(w_activity)

    st.divider()
    st.markdown("**Switch model:**  `ollama pull mistral:7b`")
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
# FAISS index cache — keyed on file hash, not just path
# (invalidates automatically when profiles.json / profiles.csv changes)
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


def _render_candidate_card(i: int, c: dict, expanded: bool = False) -> None:
    cid    = c.get("candidate_id") or c.get("id") or f"#{i}"
    title  = c.get("title", "Unknown")
    s100   = c.get("score_100", 0.0)
    llm    = c.get("llm_score",  "…")
    conf   = str(c.get("confidence", "…"))
    reason = c.get("reason", "Scoring in progress…")
    is_gem = c.get("hidden_gem", False)
    ev     = c.get("skill_evidence", {})

    gem_badge = " ★ Hidden Gem" if is_gem else ""
    label = f"**#{i}  {cid}**  |  {title}{gem_badge}  |  Score: **{s100:.1f}/100**"

    with st.expander(label, expanded=expanded):
        cols = st.columns(5)
        cols[0].metric("Score /100",    f"{s100:.1f}")
        cols[1].metric("LLM Score",     f"{llm}/10" if isinstance(llm, int) else str(llm))
        cols[2].metric("Confidence",    f"{_conf_badge(conf)} {conf.title()}")
        cols[3].metric("Skill Match",   f"{c.get('skill_score', 0):.0%}")
        cols[4].metric("Seniority",     f"{c.get('seniority_score', 0):.0%}")

        st.markdown(f"**Reason:** {reason}")

        if c.get("analysis"):
            with st.expander("Chain-of-thought reasoning"):
                st.markdown(c["analysis"])

        ev_hit  = ev.get("required_matched", [])
        ev_miss = ev.get("required_missing", [])
        if ev_hit or ev_miss:
            h_col, m_col = st.columns(2)
            h_col.markdown("✅ **Skills matched:** " + (", ".join(ev_hit) or "none"))
            m_col.markdown("❌ **Skills missing:** " + (", ".join(ev_miss) or "none"))


def _render_shortlist(ranked: list, top_n: int = 20) -> None:
    gems = [c for c in ranked if c.get("hidden_gem")]
    if gems:
        st.success(f"★ {len(gems)} Hidden Gem(s) surfaced so far")
    for i, c in enumerate(ranked[:top_n], 1):
        _render_candidate_card(i, c, expanded=(i <= 3))


# ---------------------------------------------------------------------------
# Analytics charts (shown after streaming completes)
# ---------------------------------------------------------------------------

def _show_score_breakdown(ranked: list) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("Install plotly for interactive charts: `pip install plotly`")
        return

    top = ranked[:10]
    ids = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(top)]

    signals = {
        "Embedding":  [c.get("embedding_score",  0) for c in top],
        "Skill":      [c.get("skill_score",       0) for c in top],
        "Seniority":  [c.get("seniority_score",   0) for c in top],
        "Activity":   [c.get("activity_score",    0) for c in top],
    }
    colors = ["#4e9af1", "#5cb85c", "#f0ad4e", "#d9534f"]

    fig = go.Figure()
    for (name, vals), color in zip(signals.items(), colors):
        fig.add_trace(go.Bar(name=name, x=ids, y=vals, marker_color=color))

    # Overlay composite score as a line
    composites = [c.get("composite_score", 0) for c in top]
    fig.add_trace(go.Scatter(
        name="Composite", x=ids, y=composites,
        mode="lines+markers",
        line=dict(color="white", width=2, dash="dot"),
        marker=dict(size=8),
    ))

    fig.update_layout(
        title="Signal Breakdown — Top 10 Candidates",
        barmode="group",
        xaxis_title="Candidate",
        yaxis_title="Score (0–1)",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_score_trend(ranked: list) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    scores = [c.get("score_100", 0) for c in ranked]
    ranks  = list(range(1, len(scores) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ranks, y=scores, mode="lines+markers",
        line=dict(color="#4e9af1", width=2),
        marker=dict(size=6),
        fill="tozeroy", fillcolor="rgba(78,154,241,0.15)",
    ))
    fig.update_layout(
        title="Score Dropoff Across All Ranked Candidates",
        xaxis_title="Rank",
        yaxis_title="Score /100",
        template="plotly_dark",
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_skill_gap(ranked: list, parsed_jd: dict) -> None:
    required = parsed_jd.get("required_skills", [])
    implied  = parsed_jd.get("implied_skills",  [])
    all_skills = required + implied

    if not all_skills:
        st.info("No skills extracted from JD.")
        return

    top5 = ranked[:5]
    ids  = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(top5)]

    rows = []
    for skill in all_skills:
        row = {"Skill": skill, "Type": "Required" if skill in required else "Implied"}
        for cid, c in zip(ids, top5):
            ev = c.get("skill_evidence", {})
            matched = ev.get("required_matched", []) + ev.get("implied_matched", [])
            row[cid] = "✅" if skill in matched else "❌"
        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows).set_index("Skill")

    st.dataframe(
        df.style.applymap(
            lambda v: "background-color: #1a4a1a" if v == "✅" else "background-color: #4a1a1a",
            subset=[c for c in df.columns if c != "Type"],
        ),
        use_container_width=True,
        height=min(40 * len(rows) + 80, 600),
    )


def _show_comparison(ranked: list) -> None:
    if len(ranked) < 2:
        st.info("Need at least 2 ranked candidates to compare.")
        return

    ids = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(ranked)]
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
        signals = ["Embedding", "Skill", "Seniority", "Activity", "Final (×10)"]
        a_vals  = [
            a.get("embedding_score",  0),
            a.get("skill_score",      0),
            a.get("seniority_score",  0),
            a.get("activity_score",   0),
            a.get("final_score",      0),
        ]
        b_vals  = [
            b.get("embedding_score",  0),
            b.get("skill_score",      0),
            b.get("seniority_score",  0),
            b.get("activity_score",   0),
            b.get("final_score",      0),
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(name=a_id, x=signals, y=a_vals, marker_color="#4e9af1"))
        fig.add_trace(go.Bar(name=b_id, x=signals, y=b_vals, marker_color="#f0ad4e"))
        fig.update_layout(
            title=f"{a_id} vs {b_id}",
            barmode="group",
            yaxis=dict(range=[0, 1.05]),
            template="plotly_dark",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        pass

    # Side-by-side text comparison
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

run_btn = st.button("▶  Run MANTHAN", type="primary", use_container_width=True)

if run_btn:
    if not jd_text.strip():
        st.error("Please enter a job description.")
        st.stop()

    # ── Stage 1–3 ────────────────────────────────────────────────────────────
    t0 = time.time()
    with st.status("Running Stages 1–3…", expanded=True) as status:
        st.write("Stage 1 — Parsing job description with local LLM…")
        try:
            from jd_parser import parse_jd
            parsed_jd = parse_jd(jd_text)
        except Exception as e:
            st.error(f"Stage 1 failed: {e}")
            st.stop()

        st.write("Stage 2 — FAISS recall (index cached; auto-invalidates on file change)…")
        engine, profiles = _get_index(str(data_dir), _profile_hash(data_dir))
        recalled = engine.recall(parsed_jd, top_k=min(200, len(profiles)))

        st.write("Stage 3 — Multi-signal scoring (4 signals, synonym-aware)…")
        from scoring import score_candidates
        scored = score_candidates(recalled, parsed_jd, top_n=50)
        status.update(label="Stages 1–3 done  →  starting LLM rerank", state="running")

    # ── JD summary ────────────────────────────────────────────────────────────
    st.subheader("📋 Parsed Job Requirements")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Seniority",       parsed_jd.get("seniority", "—").title())
    m2.metric("Required skills", len(parsed_jd.get("required_skills", [])))
    m3.metric("Implied skills",  len(parsed_jd.get("implied_skills",  [])))
    m4.metric("Candidates scored", len(scored))
    with st.expander("Full parsed JD"):
        st.json(parsed_jd)

    # ── Stage 4: streaming rerank ─────────────────────────────────────────────
    from rerank import rerank_stream

    st.subheader("🏆 Live Rankings — updating as model scores each candidate")
    progress   = st.progress(0, text="Starting LLM rerank…")
    results_ph = st.empty()

    ranked = []
    total  = min(rerank_n, len(scored))

    for i, result in enumerate(rerank_stream(scored, parsed_jd, top_n=rerank_n), 1):
        ranked.append(result)
        ranked_sorted = sorted(ranked, key=lambda x: x["final_score"], reverse=True)
        ranked_sorted = normalize_scores(ranked_sorted)

        progress.progress(i / total, text=f"Scored {i}/{total} candidates…")

        with results_ph.container():
            _render_shortlist(ranked_sorted, top_n=20)

    progress.progress(1.0, text=f"✅ Done in {time.time()-t0:.0f}s  —  {len(ranked_sorted)} candidates ranked")

    # ── Analytics (shown after streaming) ─────────────────────────────────────
    st.divider()
    st.subheader("📊 Analytics")

    tab_breakdown, tab_trend, tab_gap, tab_compare = st.tabs([
        "Score Breakdown", "Score Trend", "Skill Gap", "Compare",
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

    # ── Download ──────────────────────────────────────────────────────────────
    st.divider()
    st.download_button(
        "⬇  Download ranked_output.csv",
        data=_make_csv(ranked_sorted),
        file_name="ranked_output.csv",
        mime="text/csv",
        use_container_width=True,
    )
