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

from output import normalize_scores  # noqa: E402

# ---------------------------------------------------------------------------
# Paths (defined early — used in sidebar upload handler and later sections)
# ---------------------------------------------------------------------------

data_dir = Path(__file__).parent.parent / "data"

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp {
  background: radial-gradient(ellipse at top left, #130a2e 0%, #07070f 45%, #081628 100%);
  min-height: 100vh;
}

/* ── Hero banner ──────────────────────────────────────── */
.vk-hero {
  background: linear-gradient(135deg,rgba(124,58,237,.18) 0%,rgba(59,130,246,.12) 60%,rgba(16,185,129,.08) 100%);
  border: 1px solid rgba(124,58,237,.35);
  border-radius: 20px; padding: 22px 28px; margin-bottom: 20px;
}

/* ── Stat cards ───────────────────────────────────────── */
.vk-stat {
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 14px; padding: 16px 10px; text-align: center;
}
.vk-stat-n  { font-size: 30px; font-weight: 800; }
.vk-stat-l  { font-size: 11px; color: rgba(255,255,255,.45); margin-top: 3px; }
.vk-gold    { color: #fbbf24; }
.vk-purple  { color: #a78bfa; }
.vk-green   { color: #34d399; }
.vk-blue    { color: #60a5fa; }

/* ── Glass candidate card ─────────────────────────────── */
.vk-card {
  background: rgba(255,255,255,.04);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 16px; padding: 18px 20px; margin-bottom: 12px;
  transition: border-color .25s;
}
.vk-card:hover { border-color: rgba(124,58,237,.5); }

/* ── Score bar ────────────────────────────────────────── */
.vk-bar-wrap  { background:rgba(255,255,255,.07); border-radius:999px; height:7px; overflow:hidden; margin:5px 0; }
.vk-bar-fill  { height:100%; border-radius:999px; }
.vk-bar-final { background: linear-gradient(90deg,#7c3aed,#3b82f6,#10b981); }
.vk-bar-skill { background: linear-gradient(90deg,#10b981,#34d399); }
.vk-bar-embed { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
.vk-bar-ghost { background: linear-gradient(90deg,#7c3aed,#a78bfa); }

/* ── Badges ───────────────────────────────────────────── */
.vk-rank   { display:inline-flex;align-items:center;justify-content:center;width:34px;height:34px;border-radius:50%;font-weight:800;font-size:13px; }
.vk-r1     { background:linear-gradient(135deg,#f59e0b,#fcd34d);color:#000; }
.vk-r2     { background:linear-gradient(135deg,#6b7280,#d1d5db);color:#000; }
.vk-r3     { background:linear-gradient(135deg,#92400e,#d97706);color:#fff; }
.vk-rn     { background:rgba(255,255,255,.12);color:#e5e7eb; }
.vk-gem    { background:linear-gradient(135deg,#f59e0b,#ef4444);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:700;font-size:12px; }
.vk-ghost  { background:rgba(139,92,246,.2);border:1px solid rgba(139,92,246,.4);border-radius:8px;padding:3px 10px;font-size:11px;color:#a78bfa; }
.vk-dh     { background:rgba(59,130,246,.2);border:1px solid rgba(59,130,246,.4);border-radius:8px;padding:3px 10px;font-size:11px;color:#60a5fa; }
.vk-vol-hi { background:rgba(239,68,68,.2);border:1px solid rgba(239,68,68,.4);border-radius:8px;padding:3px 10px;font-size:11px;color:#f87171; }

/* ── Confidence pills ─────────────────────────────────── */
.vk-ch { background:rgba(16,185,129,.18);color:#34d399;border:1px solid rgba(16,185,129,.35);border-radius:999px;padding:2px 10px;font-size:11px;font-weight:600; }
.vk-cm { background:rgba(245,158,11,.18);color:#fbbf24;border:1px solid rgba(245,158,11,.35);border-radius:999px;padding:2px 10px;font-size:11px;font-weight:600; }
.vk-cl { background:rgba(239,68,68,.18);color:#f87171;border:1px solid rgba(239,68,68,.35);border-radius:999px;padding:2px 10px;font-size:11px;font-weight:600; }

/* ── Podium ───────────────────────────────────────────── */
.vk-pod1 { background:linear-gradient(180deg,rgba(245,158,11,.28),rgba(0,0,0,.05));border:1px solid rgba(245,158,11,.5);border-radius:16px 16px 0 0;padding:20px 16px;text-align:center;min-height:210px;display:flex;flex-direction:column;align-items:center;justify-content:center; }
.vk-pod2 { background:linear-gradient(180deg,rgba(107,114,128,.25),rgba(0,0,0,.05));border:1px solid rgba(107,114,128,.45);border-radius:16px 16px 0 0;padding:20px 16px;text-align:center;min-height:165px;display:flex;flex-direction:column;align-items:center;justify-content:center;margin-top:45px; }
.vk-pod3 { background:linear-gradient(180deg,rgba(146,64,14,.25),rgba(0,0,0,.05));border:1px solid rgba(146,64,14,.45);border-radius:16px 16px 0 0;padding:20px 16px;text-align:center;min-height:135px;display:flex;flex-direction:column;align-items:center;justify-content:center;margin-top:75px; }
.vk-pod-score { font-size:30px;font-weight:800; }
.vk-pod-id    { font-size:13px;font-weight:700;margin:4px 0; }
.vk-pod-title { font-size:11px;color:rgba(255,255,255,.55);margin-bottom:6px; }

/* ── Section headers ─────────────────────────────────── */
.vk-section { font-size:17px;font-weight:700;
  background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  margin:8px 0 14px;
}

/* ── Animated gradient text ──────────────────────────── */
@keyframes vk-grad { 0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%} }
.vk-animated {
  background:linear-gradient(270deg,#7c3aed,#3b82f6,#10b981,#f59e0b);
  background-size:400% 400%;
  animation:vk-grad 8s ease infinite;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  font-weight:800;
}

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:rgba(255,255,255,.02); }
::-webkit-scrollbar-thumb { background:rgba(124,58,237,.5);border-radius:3px; }

/* ── Input fields ────────────────────────────────────── */
.stTextArea textarea,.stTextInput input {
  background:rgba(255,255,255,.04)!important;
  border:1px solid rgba(255,255,255,.1)!important;
  border-radius:12px!important; color:#e2e8f0!important;
}
.stButton>button[kind="primary"] {
  background:linear-gradient(135deg,#7c3aed,#3b82f6)!important;
  border:none!important; border-radius:12px!important;
  font-weight:700!important; letter-spacing:.3px!important;
}
.streamlit-expanderHeader { border-radius:12px!important; }
</style>
""", unsafe_allow_html=True)

_provider = os.getenv("LLM_PROVIDER", "ollama").lower()

st.markdown("""
<div class="vk-hero">
  <div class="vk-animated" style="font-size:28px;margin-bottom:4px">🔍 VIVEKA</div>
  <div style="color:rgba(255,255,255,.7);font-size:15px;font-weight:500">
    Discerning True Talent from Noise &nbsp;·&nbsp;
    <span style="color:#a78bfa">AI-Powered Candidate Ranking</span>
  </div>
</div>
""", unsafe_allow_html=True)

if _provider == "gemini":
    st.caption("☁️ Cloud Demo · Powered by Gemini · Free tier")
else:
    st.caption("🖥️ Offline · Free · Private · Powered by Ollama locally")

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    if _provider == "gemini":
        st.info("☁️ Cloud mode — using Gemini API")
    else:
        model_name = st.text_input(
            "Ollama model",
            value=os.getenv("VIVEKA_MODEL", "llama3.2"),
            help="Must be pulled: `ollama pull llama3.2`",
        )
        os.environ["VIVEKA_MODEL"] = model_name

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
    os.environ["VIVEKA_PARALLEL_WORKERS"] = str(max_workers)

    st.divider()
    st.subheader("Signal Weights")
    st.caption("Auto-normalised. Drag to adjust contribution to composite score.")

    w_embed     = st.slider("Embedding (semantic)",  0.0, 1.0, 0.30, 0.05)
    w_skill     = st.slider("Skill match",           0.0, 1.0, 0.40, 0.05)
    w_seniority = st.slider("Seniority fit",         0.0, 1.0, 0.15, 0.05)
    w_activity  = st.slider("Activity / behavior",   0.0, 1.0, 0.15, 0.05)

    os.environ["VIVEKA_W_EMBED"]     = str(w_embed)
    os.environ["VIVEKA_W_SKILL"]     = str(w_skill)
    os.environ["VIVEKA_W_SENIORITY"] = str(w_seniority)
    os.environ["VIVEKA_W_ACTIVITY"]  = str(w_activity)

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
    os.environ["VIVEKA_PII_FIREWALL"] = "on" if pii_on else "off"
    if pii_on:
        st.success("PII firewall ON — identity fields stripped before scoring.")
    else:
        st.warning("PII firewall OFF — identity fields visible to model.")

    audit_on = st.toggle(
        "Audit trail (log decisions to audit.jsonl)",
        value=True,
        help="Writes every ranking decision to data/audit.jsonl for reproducibility.",
    )
    os.environ["VIVEKA_AUDIT"] = "on" if audit_on else "off"

    if _provider == "ollama":
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


def _conf_pill(conf: str) -> str:
    cls = {"high": "vk-ch", "medium": "vk-cm", "low": "vk-cl"}.get(str(conf).lower(), "vk-cm")
    return f'<span class="{cls}">{conf.title()}</span>'


def _rank_badge(i: int) -> str:
    cls = {1: "vk-r1", 2: "vk-r2", 3: "vk-r3"}.get(i, "vk-rn")
    return f'<span class="vk-rank {cls}">#{i}</span>'


def _score_bar(pct: float, cls: str = "vk-bar-final") -> str:
    w = max(0, min(100, pct * 100))
    return (
        f'<div class="vk-bar-wrap">'
        f'<div class="vk-bar-fill {cls}" style="width:{w:.1f}%"></div>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Ghost Intelligence — forward-looking signals
# ---------------------------------------------------------------------------

def compute_ghost_scores(ranked: list) -> list:
    """Enrich each candidate with ghost_score, volatility, dark_horse flag."""
    result = []
    for c in ranked:
        c = dict(c)
        composite = c.get("composite_score", 0)
        activity  = c.get("activity_score",  0)
        is_gem    = c.get("hidden_gem",       False)
        stuffing  = c.get("stuffing", {}).get("stuffing_ratio", 0)
        cal_conf  = c.get("calibrated_confidence", 0.5)
        embed     = c.get("embedding_score",  0)
        skill     = c.get("skill_score",      0)

        # Ghost = potential-adjusted score (activity uplift + gem bonus - stuffing drag)
        ghost = min(1.0, composite + activity * 0.12 + (0.06 if is_gem else 0) - stuffing * 0.08)

        # Volatility = how uncertain is the ranking
        volatility = round(stuffing * (1.0 - cal_conf), 4)

        # Dark horse = low embedding match but strong skill coverage
        dark_horse = embed < 0.45 and skill > 0.55

        c["ghost_score"] = round(ghost, 4)
        c["volatility"]  = volatility
        c["dark_horse"]  = dark_horse
        result.append(c)
    return result


# ---------------------------------------------------------------------------
# Podium — top 3 showcase
# ---------------------------------------------------------------------------

def _render_podium(ranked: list) -> None:
    if len(ranked) < 1:
        return
    top3 = ranked[:3]

    medals      = ["🥇", "🥈", "🥉"]
    pod_classes = ["vk-pod1", "vk-pod2", "vk-pod3"]
    score_cols  = ["#fbbf24", "#d1d5db", "#d97706"]

    # Podium order: 2nd | 1st | 3rd
    order = [1, 0, 2] if len(top3) == 3 else [0] if len(top3) == 1 else [1, 0]
    cols  = st.columns(len(order))

    for col_idx, rank_idx in enumerate(order):
        if rank_idx >= len(top3):
            continue
        c    = top3[rank_idx]
        cid  = c.get("candidate_id") or c.get("id") or f"#{rank_idx+1}"
        titl = (c.get("title") or "")[:24]
        s100 = c.get("score_100", 0)
        gh   = c.get("ghost_score", 0)
        gem  = "⭐ " if c.get("hidden_gem") else ""
        dh   = "💎 " if c.get("dark_horse") else ""
        cls  = pod_classes[rank_idx]
        col  = score_cols[rank_idx]

        cols[col_idx].markdown(
            f'<div class="{cls}">'
            f'<div style="font-size:38px">{medals[rank_idx]}</div>'
            f'<div class="vk-pod-id" style="color:{col}">{gem}{dh}{cid}</div>'
            f'<div class="vk-pod-title">{titl}</div>'
            f'<div class="vk-pod-score" style="color:{col}">{s100:.0f}</div>'
            f'<div style="font-size:10px;color:rgba(255,255,255,.4)">/100</div>'
            f'<div style="margin-top:8px"><span class="vk-ghost">Ghost {gh*100:.0f}</span></div>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Ghost Intelligence analytics tab
# ---------------------------------------------------------------------------

def _show_ghost_intel(ranked: list) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("Install plotly for ghost intelligence charts.")
        return

    ids    = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(ranked)]
    actual = [c.get("final_score",  0) for c in ranked]
    ghost  = [c.get("ghost_score",  0) for c in ranked]
    volat  = [c.get("volatility",   0) for c in ranked]

    # ── Ghost vs Actual scatter ────────────────────────────────────────────────
    marker_colors  = []
    marker_symbols = []
    hover_texts    = []
    for i, c in enumerate(ranked):
        is_gem = c.get("hidden_gem", False)
        is_dh  = c.get("dark_horse", False)
        marker_colors.append("#f59e0b" if is_gem else ("#a78bfa" if is_dh else "#60a5fa"))
        marker_symbols.append("star" if is_gem else ("diamond" if is_dh else "circle"))
        hover_texts.append(
            f"<b>{ids[i]}</b><br>Actual: {actual[i]:.3f}<br>"
            f"Ghost: {ghost[i]:.3f}<br>Volatility: {volat[i]:.3f}"
        )

    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="rgba(255,255,255,0.2)", dash="dot"))
    fig.add_annotation(x=0.75, y=0.85, text="Ghost > Actual = undervalued",
                       showarrow=False, font=dict(color="rgba(255,255,255,0.4)", size=11))

    fig.add_trace(go.Scatter(
        x=actual, y=ghost,
        mode="markers+text",
        text=ids,
        textposition="top center",
        textfont=dict(size=10),
        marker=dict(size=14, color=marker_colors, symbol=marker_symbols,
                    line=dict(width=1, color="rgba(255,255,255,0.3)")),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
        showlegend=False,
    ))
    fig.update_layout(
        title="Ghost Score vs Actual Score",
        xaxis=dict(title="Actual Final Score", range=[0, 1.05]),
        yaxis=dict(title="Ghost Score (Future Potential)", range=[0, 1.05]),
        template="plotly_dark", height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown("🌟 **Star** = Hidden Gem")
    c2.markdown("💎 **Diamond** = Dark Horse (low embed, high skills)")
    c3.markdown("🔵 **Circle** = Standard candidate")

    # ── Volatility bar ────────────────────────────────────────────────────────
    st.markdown("#### Ranking Volatility — higher means less certain")
    top_vol = sorted(zip(ids, volat), key=lambda x: x[1], reverse=True)[:12]
    v_ids   = [x[0] for x in top_vol]
    v_vals  = [x[1] for x in top_vol]
    bar_col = ["#ef4444" if v > 0.3 else ("#f59e0b" if v > 0.1 else "#10b981") for v in v_vals]

    fig2 = go.Figure(go.Bar(x=v_ids, y=v_vals, marker_color=bar_col,
                             text=[f"{v:.2f}" for v in v_vals], textposition="outside"))
    fig2.update_layout(yaxis_title="Volatility Index", xaxis_tickangle=-30,
                       template="plotly_dark", height=320,
                       yaxis=dict(range=[0, max(v_vals or [0.1]) * 1.3]))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("🔴 > 0.30: unstable ranking  |  🟡 0.10–0.30: moderate  |  🟢 < 0.10: very stable")

    # ── Ghost uplift table ────────────────────────────────────────────────────
    st.markdown("#### Ghost Uplift — candidates with highest hidden potential")
    import pandas as pd
    rows = []
    for i, c in enumerate(ranked):
        cid  = ids[i]
        up   = c.get("ghost_score", 0) - c.get("final_score", 0)
        rows.append({
            "Rank":       i + 1,
            "Candidate":  cid,
            "Title":      (c.get("title") or "")[:22],
            "Actual /100": f"{c.get('score_100', 0):.1f}",
            "Ghost /100":  f"{c.get('ghost_score', 0)*100:.1f}",
            "Uplift":      f"+{up*100:.1f}" if up > 0 else f"{up*100:.1f}",
            "Dark Horse":  "💎" if c.get("dark_horse") else "",
            "Hidden Gem":  "⭐" if c.get("hidden_gem") else "",
        })
    rows.sort(key=lambda r: float(r["Uplift"]), reverse=True)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=350)


# ---------------------------------------------------------------------------
# What-If Simulator
# ---------------------------------------------------------------------------

def _show_whatif(ranked: list, parsed_jd: dict) -> None:
    required = parsed_jd.get("required_skills", [])
    implied  = parsed_jd.get("implied_skills",  [])

    ids = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(ranked)]
    sel = st.selectbox("Choose a candidate to simulate", ids, key="wi_sel")
    c   = ranked[ids.index(sel)]
    ev  = c.get("skill_evidence", {})

    miss_req  = ev.get("required_missing", [])
    miss_impl = ev.get("implied_missing",  [])

    if not miss_req and not miss_impl:
        st.success(f"✅ {sel} already matches all required and implied skills — perfect candidate!")
        return

    st.markdown("**Hypothetically add skills and see the rank impact:**")

    added_req, added_impl = [], []
    if miss_req:
        st.markdown("🔴 **Missing required skills**")
        req_cols = st.columns(min(4, len(miss_req)))
        for i, skill in enumerate(miss_req):
            if req_cols[i % len(req_cols)].checkbox(skill, key=f"wi_r_{sel}_{skill}"):
                added_req.append(skill)

    if miss_impl:
        st.markdown("🟡 **Missing implied skills**")
        imp_cols = st.columns(min(4, len(miss_impl)))
        for i, skill in enumerate(miss_impl):
            if imp_cols[i % len(imp_cols)].checkbox(skill, key=f"wi_i_{sel}_{skill}"):
                added_impl.append(skill)

    if added_req or added_impl:
        old_req_hit  = len(ev.get("required_matched", []))
        old_impl_hit = len(ev.get("implied_matched",  []))
        new_req_hit  = old_req_hit  + len(added_req)
        new_impl_hit = old_impl_hit + len(added_impl)

        max_score    = len(required) * 1.0 + len(implied) * 0.5 or 1.0
        old_raw      = (old_req_hit * 1.0 + old_impl_hit * 0.5) / max_score
        new_raw      = (new_req_hit * 1.0 + new_impl_hit * 0.5) / max_score

        stuffing     = c.get("stuffing", {}).get("stuffing_ratio", 0)
        try:
            from config import STUFFING_PENALTY
        except ImportError:
            STUFFING_PENALTY = 0.3

        old_skill = old_raw * (1 - stuffing * STUFFING_PENALTY)
        new_skill = new_raw * (1 - stuffing * STUFFING_PENALTY)
        delta_skill = new_skill - old_skill

        try:
            from config import get_weights
            _, w_skill, _, _ = get_weights()
        except ImportError:
            w_skill = 0.4

        comp_delta = delta_skill * w_skill

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Skill Score",    f"{old_skill:.0%}")
        col2.metric("New Skill Score",        f"{new_skill:.0%}",  f"+{delta_skill:.0%}")
        col3.metric("Composite Impact",       f"+{comp_delta:.3f}")

        # Hypothetical rank
        hypo = [(c2.get("composite_score", 0), ids[j])
                for j, c2 in enumerate(ranked) if ids[j] != sel]
        hypo.append((c.get("composite_score", 0) + comp_delta, sel))
        hypo.sort(reverse=True)
        new_rank = next(j for j, (_, cid) in enumerate(hypo, 1) if cid == sel)
        old_rank = ids.index(sel) + 1
        rank_delta = old_rank - new_rank

        col4.metric(
            "Hypothetical Rank",
            f"#{new_rank}",
            (f"↑ {rank_delta} places" if rank_delta > 0
             else ("No change" if rank_delta == 0 else f"↓ {abs(rank_delta)} places")),
        )

        st.info(
            f"Adding **{', '.join(added_req + added_impl)}** would move "
            f"**{sel}** from rank #{old_rank} → #{new_rank}"
        )


def _render_candidate_card(i: int, c: dict, expanded: bool = False, render_id: int = 0) -> None:
    cid    = c.get("candidate_id") or c.get("id") or f"#{i}"
    title  = c.get("title", "Unknown")
    s100   = c.get("score_100", 0.0)
    llm    = c.get("llm_score",  "…")
    conf   = str(c.get("confidence", "…"))
    cal_c  = c.get("calibrated_confidence", 0.5)
    reason = c.get("reason", "Scoring in progress…")
    is_gem = c.get("hidden_gem", False)
    is_dh  = c.get("dark_horse", False)
    ghost  = c.get("ghost_score", 0.0)
    volat  = c.get("volatility",  0.0)
    ev       = c.get("skill_evidence", {})
    stuffing = c.get("stuffing", {})
    cf       = c.get("counterfactual", {})

    badges = ""
    if is_gem:
        badges += ' <span class="vk-gem">⭐ Hidden Gem</span>'
    if is_dh:
        badges += ' <span class="vk-dh">💎 Dark Horse</span>'
    if volat > 0.3:
        badges += ' <span class="vk-vol-hi">⚡ Volatile</span>'

    label = (
        f"{_rank_badge(i)} &nbsp; **{cid}** &nbsp;·&nbsp; {title}"
        f"{badges} &nbsp;·&nbsp; Score: **{s100:.1f}/100**"
    )

    with st.expander(label, expanded=expanded):
        # ── Score bar strip ────────────────────────────────────────────────
        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:14px;margin-bottom:10px'>"
            # Final score
            f"<div><div style='font-size:11px;color:rgba(255,255,255,.5)'>Final Score</div>"
            f"<div style='font-size:22px;font-weight:800;color:#a78bfa'>{s100:.0f}<span style='font-size:13px;color:rgba(255,255,255,.4)'>/100</span></div>"
            f"{_score_bar(s100/100,'vk-bar-final')}</div>"
            # Skill
            f"<div><div style='font-size:11px;color:rgba(255,255,255,.5)'>Skill Match</div>"
            f"<div style='font-size:22px;font-weight:800;color:#34d399'>{c.get('skill_score',0):.0%}</div>"
            f"{_score_bar(c.get('skill_score',0),'vk-bar-skill')}</div>"
            # Embedding
            f"<div><div style='font-size:11px;color:rgba(255,255,255,.5)'>Semantic</div>"
            f"<div style='font-size:22px;font-weight:800;color:#60a5fa'>{c.get('embedding_score',0):.0%}</div>"
            f"{_score_bar(c.get('embedding_score',0),'vk-bar-embed')}</div>"
            # Ghost
            f"<div><div style='font-size:11px;color:rgba(255,255,255,.5)'>Ghost Score</div>"
            f"<div style='font-size:22px;font-weight:800;color:#c084fc'>{ghost*100:.0f}<span style='font-size:13px;color:rgba(255,255,255,.4)'>/100</span></div>"
            f"{_score_bar(ghost,'vk-bar-ghost')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Second row: LLM score, confidence, seniority, cal.conf ────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("LLM Score",   f"{llm}/10" if isinstance(llm, int) else str(llm))
        col2.metric("Seniority",   f"{c.get('seniority_score',0):.0%}")
        col3.metric("Activity",    f"{c.get('activity_score',0):.0%}")
        col4.metric("Cal. Conf",   f"{cal_c:.0%}")

        # Confidence pill + reason
        st.markdown(
            f"{_conf_pill(conf)} &nbsp; {reason}",
            unsafe_allow_html=True,
        )

        # ── Citation evidence ──────────────────────────────────────────────
        ev_verified    = c.get("evidence_verified",    [])
        ev_unsupported = c.get("evidence_unsupported", [])
        if ev_verified or ev_unsupported:
            with st.expander("📎 Citation Evidence"):
                for snippet in ev_verified:
                    st.success(f'✅ **Verified:** "{snippet}"')
                for snippet in ev_unsupported:
                    st.error(f'⚠️ **Not found in profile:** "{snippet}"')

        if c.get("analysis"):
            with st.expander("🧠 Chain-of-thought reasoning"):
                st.markdown(c["analysis"])

        # ── Skill evidence ─────────────────────────────────────────────────
        ev_hit  = ev.get("required_matched", [])
        ev_miss = ev.get("required_missing", [])
        if ev_hit or ev_miss:
            h_col, m_col = st.columns(2)
            h_col.markdown("✅ **Matched:** " + (", ".join(ev_hit) or "none"))
            m_col.markdown("❌ **Missing:** " + (", ".join(ev_miss) or "none"))

        if stuffing and stuffing.get("stuffing_ratio", 0) > 0.2:
            ratio = stuffing.get("stuffing_ratio", 0)
            unsup = stuffing.get("claimed_unsupported", [])
            st.warning(
                f"⚠️ **Résumé stuffing {ratio:.0%}** — claims with no narrative support: "
                + ", ".join(unsup[:5])
            )

        if cf and cf.get("ceiling"):
            st.info(f"🔮 **Why not higher?** {cf['ceiling']}")

        # ── Radar chart ────────────────────────────────────────────────────
        try:
            import plotly.graph_objects as go
            r_vals = [
                c.get("embedding_score",  0),
                c.get("skill_score",      0),
                c.get("seniority_score",  0),
                c.get("activity_score",   0),
                (c.get("llm_score", 0) or 0) / 10.0,
                cal_c,
                ghost,
            ]
            theta = ["Semantic", "Skills", "Seniority", "Activity", "LLM Fit", "Cal.Conf", "Ghost"]
            fig = go.Figure(go.Scatterpolar(
                r=r_vals + [r_vals[0]],
                theta=theta + [theta[0]],
                fill="toself",
                fillcolor="rgba(124,58,237,0.18)",
                line=dict(color="#a78bfa", width=2),
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False,
                                           gridcolor="rgba(255,255,255,0.08)"),
                           angularaxis=dict(color="rgba(255,255,255,0.5)")),
                showlegend=False,
                height=260,
                margin=dict(l=30, r=30, t=30, b=30),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True, key=f"radar_{render_id}_{i}_{cid[:8]}")
        except ImportError:
            pass


def _render_shortlist(ranked: list, top_n: int = 20, render_id: int = 0) -> None:
    gems    = [c for c in ranked if c.get("hidden_gem")]
    dh_list = [c for c in ranked if c.get("dark_horse")]
    if gems or dh_list:
        msg = []
        if gems:
            msg.append(f"⭐ {len(gems)} Hidden Gem(s)")
        if dh_list:
            msg.append(f"💎 {len(dh_list)} Dark Horse(s)")
        st.success("  ·  ".join(msg) + " surfaced")

    if len(ranked) >= 2 and render_id == 0:
        _render_podium(ranked)
        st.markdown("---")

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
    def style_fn(v):
        return "background-color: #1a4a1a" if v == "✅" else "background-color: #4a1a1a"
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

    # ── Ollama health check (skip in Gemini/cloud mode) ───────────────────────
    if _provider == "ollama":
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
            _render_shortlist(ranked_sorted, top_n=20, render_id=i if i < total else 0)

    t_total = time.time() - t0
    progress.progress(1.0, text=f"✅ Done in {t_total:.0f}s  —  {len(ranked_sorted)} candidates ranked")
    eta_ph.empty()

    # Enrich with ghost scores + volatility + dark horse flags
    ranked_sorted = compute_ghost_scores(ranked_sorted)

    # ── Summary stat cards ─────────────────────────────────────────────────────
    n_gems  = sum(1 for c in ranked_sorted if c.get("hidden_gem"))
    n_dh    = sum(1 for c in ranked_sorted if c.get("dark_horse"))
    top_g   = ranked_sorted[0].get("ghost_score", 0) if ranked_sorted else 0
    avg_vol = sum(c.get("volatility", 0) for c in ranked_sorted) / max(len(ranked_sorted), 1)

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.markdown(f'<div class="vk-stat"><div class="vk-stat-n vk-purple">{len(ranked_sorted)}</div><div class="vk-stat-l">Ranked</div></div>', unsafe_allow_html=True)
    sc2.markdown(f'<div class="vk-stat"><div class="vk-stat-n vk-gold">{n_gems}</div><div class="vk-stat-l">Hidden Gems</div></div>', unsafe_allow_html=True)
    sc3.markdown(f'<div class="vk-stat"><div class="vk-stat-n vk-blue">{n_dh}</div><div class="vk-stat-l">Dark Horses</div></div>', unsafe_allow_html=True)
    sc4.markdown(f'<div class="vk-stat"><div class="vk-stat-n vk-purple">{top_g*100:.0f}</div><div class="vk-stat-l">Top Ghost /100</div></div>', unsafe_allow_html=True)
    sc5.markdown(f'<div class="vk-stat"><div class="vk-stat-n vk-green">{avg_vol:.2f}</div><div class="vk-stat-l">Avg Volatility</div></div>', unsafe_allow_html=True)

    # ── Analytics ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Analytics")

    tab_ghost, tab_whatif, tab_breakdown, tab_trend, tab_gap, tab_compare, tab_stuffing, tab_cite = st.tabs([
        "👻 Ghost Intel", "🔮 What-If",
        "Score Breakdown", "Score Trend", "Skill Gap", "Compare",
        "Stuffing Detector", "Citation Audit",
    ])
    with tab_ghost:
        st.caption("Forward-looking signals: ghost scores, volatility, dark horses, and hidden uplift potential.")
        _show_ghost_intel(ranked_sorted)
    with tab_whatif:
        st.caption("Add missing skills to any candidate and see the hypothetical rank change — instantly.")
        _show_whatif(ranked_sorted, parsed_jd)
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
