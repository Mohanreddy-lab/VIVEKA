"""
gradio_app.py — VIVEKA God-Level Gradio UI

Replaces Streamlit with a native Gradio Blocks interface:
  • Animated glassmorphism design system
  • Streaming live rankings (generator-based)
  • Podium view for top-3
  • Ghost Score + Dark Horse + Volatility badges
  • Score bar strips per candidate
  • Ghost Intelligence tab (scatter, volatility, uplift table)
  • What-If Simulator (add skills → instant rank change)
  • Full analytics suite (breakdown, trend, skill gap, compare)
  • One-click CSV/JSON download
"""

import json
import os
import sys
import time
from pathlib import Path

import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))

_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
_data_dir  = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# CSS — glassmorphism dark design system
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .gradio-container, [class*="svelte"] {
    font-family: 'Inter', sans-serif !important;
}
.gradio-container {
    background: radial-gradient(ellipse at top left,#130a2e 0%,#07070f 45%,#081628 100%) !important;
    max-width: 100% !important; min-height: 100vh;
}
.main { background: transparent !important; }

/* ── Hero ──────────────────────────────────────────────────── */
.vk-hero {
    background: linear-gradient(135deg,rgba(124,58,237,.2) 0%,rgba(59,130,246,.13) 60%,rgba(16,185,129,.08) 100%);
    border: 1px solid rgba(124,58,237,.4); border-radius: 22px;
    padding: 30px 40px; margin-bottom: 24px; text-align: center;
    position: relative; overflow: hidden;
}
@keyframes vk-grad {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}
.vk-logo {
    font-size: 38px; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(270deg,#7c3aed,#3b82f6,#10b981,#f59e0b);
    background-size: 400% 400%; animation: vk-grad 8s ease infinite;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.vk-tagline { font-size: 16px; font-weight: 600; color: rgba(255,255,255,.85); }
.vk-sub     { font-size: 13px; color: rgba(255,255,255,.45); margin-top: 4px; }

/* ── Provider badge ───────────────────────────────────────── */
.provider-badge {
    display: inline-block; padding: 6px 14px; border-radius: 999px;
    font-size: 12px; font-weight: 600; margin-top: 10px;
    background: rgba(124,58,237,.18); border: 1px solid rgba(124,58,237,.4);
    color: #a78bfa;
}

/* ── Stat strip ───────────────────────────────────────────── */
.vk-stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px; }
.vk-stat {
    flex: 1; min-width: 90px; background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.09); border-radius: 14px;
    padding: 14px 10px; text-align: center;
}
.vk-stat-n { font-size: 26px; font-weight: 800; }
.vk-stat-l { font-size: 10px; color: rgba(255,255,255,.4); margin-top: 2px; }
.c-purple { color: #a78bfa; } .c-gold { color: #fbbf24; }
.c-blue   { color: #60a5fa; } .c-green { color: #34d399; }

/* ── Podium ───────────────────────────────────────────────── */
.podium { display: flex; justify-content: center; align-items: flex-end; gap: 10px; margin: 24px 0; }
.pod-col { flex: 1; max-width: 220px; text-align: center; }
.pod-base {
    border-radius: 16px 16px 0 0; padding: 20px 14px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.pod-1 { background: linear-gradient(180deg,rgba(245,158,11,.3),rgba(0,0,0,.05)); border: 1px solid rgba(245,158,11,.55); min-height: 200px; }
.pod-2 { background: linear-gradient(180deg,rgba(107,114,128,.25),rgba(0,0,0,.05)); border: 1px solid rgba(107,114,128,.45); min-height: 160px; margin-top: 40px; }
.pod-3 { background: linear-gradient(180deg,rgba(146,64,14,.25),rgba(0,0,0,.05)); border: 1px solid rgba(146,64,14,.45); min-height: 130px; margin-top: 70px; }
.pod-score { font-size: 30px; font-weight: 800; }
.pod-id    { font-size: 13px; font-weight: 700; margin: 5px 0; }
.pod-title { font-size: 11px; color: rgba(255,255,255,.5); margin-bottom: 8px; }
.pod-ghost { font-size: 11px; color: #a78bfa; margin-top: 8px; }

/* ── Candidate card ───────────────────────────────────────── */
.vk-card {
    background: rgba(255,255,255,.04); backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,.09); border-radius: 16px;
    padding: 18px 20px; margin-bottom: 10px;
    transition: border-color .2s;
}
.vk-card:hover { border-color: rgba(124,58,237,.5); }
.card-header { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.rank-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 36px; height: 36px; border-radius: 50%; font-weight: 800; font-size: 13px; flex-shrink: 0;
}
.r1 { background: linear-gradient(135deg,#f59e0b,#fcd34d); color: #000; }
.r2 { background: linear-gradient(135deg,#6b7280,#d1d5db); color: #000; }
.r3 { background: linear-gradient(135deg,#92400e,#d97706); color: #fff; }
.rn { background: rgba(255,255,255,.12); color: #e5e7eb; }
.card-title-area { flex: 1; }
.candidate-id    { font-size: 15px; font-weight: 700; color: #e2e8f0; display: block; }
.candidate-title { font-size: 12px; color: rgba(255,255,255,.5); }
.score-big { font-size: 26px; font-weight: 800; color: #a78bfa; text-align: right; white-space: nowrap; }
.score-sub { font-size: 12px; color: rgba(255,255,255,.35); }

/* ── Badges ───────────────────────────────────────────────── */
.badges { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
.badge {
    font-size: 11px; font-weight: 600; padding: 2px 9px; border-radius: 999px;
    display: inline-block;
}
.badge-gem  { background: rgba(245,158,11,.2); border: 1px solid rgba(245,158,11,.45); color: #fbbf24; }
.badge-dh   { background: rgba(59,130,246,.2);  border: 1px solid rgba(59,130,246,.45);  color: #60a5fa; }
.badge-vol  { background: rgba(239,68,68,.2);   border: 1px solid rgba(239,68,68,.45);   color: #f87171; }

/* ── Score bars ───────────────────────────────────────────── */
.score-bars { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 20px; margin-bottom: 12px; }
.bar-row { display: flex; align-items: center; gap: 8px; }
.bar-label { font-size: 10px; color: rgba(255,255,255,.45); width: 52px; flex-shrink: 0; }
.bar-track { flex: 1; background: rgba(255,255,255,.07); border-radius: 999px; height: 6px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 999px; transition: width .8s cubic-bezier(.4,0,.2,1); }
.bar-final { background: linear-gradient(90deg,#7c3aed,#3b82f6,#10b981); }
.bar-skill { background: linear-gradient(90deg,#10b981,#34d399); }
.bar-embed { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
.bar-ghost { background: linear-gradient(90deg,#7c3aed,#a78bfa); }
.bar-val   { font-size: 11px; color: rgba(255,255,255,.55); width: 32px; text-align: right; }

/* ── Confidence pills ─────────────────────────────────────── */
.conf-pill { border-radius: 999px; padding: 2px 10px; font-size: 11px; font-weight: 600; }
.c-high    { background: rgba(16,185,129,.18); border: 1px solid rgba(16,185,129,.35); color: #34d399; }
.c-medium  { background: rgba(245,158,11,.18); border: 1px solid rgba(245,158,11,.35); color: #fbbf24; }
.c-low     { background: rgba(239,68,68,.18);  border: 1px solid rgba(239,68,68,.35);  color: #f87171; }

/* ── Card reason / skills ─────────────────────────────────── */
.card-reason  { font-size: 13px; color: rgba(255,255,255,.75); line-height: 1.5; margin: 8px 0; }
.skills-row   { font-size: 12px; margin-top: 4px; }
.skill-match  { color: #34d399; }
.skill-miss   { color: #f87171; }

/* ── Status / progress ────────────────────────────────────── */
.vk-status {
    background: rgba(124,58,237,.12); border: 1px solid rgba(124,58,237,.3);
    border-radius: 10px; padding: 10px 16px; font-size: 13px; color: #c4b5fd;
    margin-bottom: 12px;
}
.vk-progress     { background: rgba(255,255,255,.07); border-radius: 999px; height: 6px; margin-bottom: 8px; overflow: hidden; }
.vk-progress-bar { background: linear-gradient(90deg,#7c3aed,#3b82f6); height: 100%; border-radius: 999px; transition: width .4s; }
.vk-error {
    background: rgba(239,68,68,.12); border: 1px solid rgba(239,68,68,.35);
    border-radius: 10px; padding: 12px 16px; color: #fca5a5; font-size: 13px;
}
.vk-jd-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
.vk-jd-pill {
    font-size: 12px; padding: 4px 12px; border-radius: 999px;
    background: rgba(59,130,246,.15); border: 1px solid rgba(59,130,246,.3); color: #93c5fd;
}

/* ── Empty state ──────────────────────────────────────────── */
.empty-state {
    text-align: center; padding: 60px 20px;
    color: rgba(255,255,255,.3); font-size: 15px; font-style: italic;
}

/* ── Stat pill strip ──────────────────────────────────────── */
.stat-pills { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
.stat-pill  {
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1);
    border-radius: 10px; padding: 6px 14px; font-size: 13px; color: rgba(255,255,255,.65);
}

/* ── Gradio overrides ─────────────────────────────────────── */
.gr-button, button[data-testid] { border-radius: 12px !important; font-weight: 600 !important; }
.primary { background: linear-gradient(135deg,#7c3aed,#3b82f6) !important; border: none !important; }
textarea, input[type=text], input[type=number] {
    background: rgba(255,255,255,.05) !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 12px !important; color: #e2e8f0 !important;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,.5); border-radius: 3px; }
"""

# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _ghost(ranked: list) -> list:
    out = []
    for c in ranked:
        c = dict(c)
        comp  = c.get("composite_score", 0)
        act   = c.get("activity_score",  0)
        gem   = c.get("hidden_gem",       False)
        stuff = c.get("stuffing", {}).get("stuffing_ratio", 0)
        cal   = c.get("calibrated_confidence", 0.5)
        emb   = c.get("embedding_score",  0)
        skill = c.get("skill_score",      0)
        c["ghost_score"] = round(min(1.0, comp + act * 0.12 + (0.06 if gem else 0) - stuff * 0.08), 4)
        c["volatility"]  = round(stuff * (1.0 - cal), 4)
        c["dark_horse"]  = emb < 0.45 and skill > 0.55
        out.append(c)
    return out


def _card_html(rank: int, c: dict) -> str:
    cid    = c.get("candidate_id") or c.get("id") or f"#{rank}"
    title  = (c.get("title") or "")[:30]
    s100   = c.get("score_100", 0.0)
    skill  = c.get("skill_score",      0)
    embed  = c.get("embedding_score",  0)
    ghost  = c.get("ghost_score",      0)
    conf   = str(c.get("confidence",   "medium")).lower()
    reason = c.get("reason", "")
    ev     = c.get("skill_evidence",   {})
    stuff  = c.get("stuffing",         {})
    is_gem = c.get("hidden_gem", False)
    is_dh  = c.get("dark_horse", False)
    volat  = c.get("volatility",  0)

    rank_cls = {1: "r1", 2: "r2", 3: "r3"}.get(rank, "rn")
    conf_cls = {"high": "c-high", "medium": "c-medium", "low": "c-low"}.get(conf, "c-medium")

    badges = ""
    if is_gem:
        badges += '<span class="badge badge-gem">⭐ Hidden Gem</span>'
    if is_dh:
        badges += '<span class="badge badge-dh">💎 Dark Horse</span>'
    if volat > 0.3:
        badges += '<span class="badge badge-vol">⚡ Volatile</span>'

    matched = ", ".join(ev.get("required_matched", []))
    missing = ", ".join(ev.get("required_missing", []))
    stuffing_ratio = stuff.get("stuffing_ratio", 0)
    stuffed_skills = ", ".join(stuff.get("claimed_unsupported", [])[:4])

    skills_html = ""
    if matched:
        skills_html += f'<div class="skills-row"><span class="skill-match">✅ {matched}</span></div>'
    if missing:
        skills_html += f'<div class="skills-row"><span class="skill-miss">❌ {missing}</span></div>'
    if stuffing_ratio > 0.2:
        skills_html += (
            f'<div class="skills-row" style="color:#fbbf24;font-size:12px">'
            f'⚠️ Stuffing {stuffing_ratio:.0%}: {stuffed_skills}</div>'
        )

    return f"""
<div class="vk-card">
  <div class="card-header">
    <span class="rank-badge {rank_cls}">#{rank}</span>
    <div class="card-title-area">
      <span class="candidate-id">{cid}</span>
      <span class="candidate-title">{title}</span>
      <div class="badges">{badges}</div>
    </div>
    <div class="score-big">{s100:.0f}<span class="score-sub">/100</span></div>
  </div>
  <div class="score-bars">
    <div class="bar-row">
      <span class="bar-label">Final</span>
      <div class="bar-track"><div class="bar-fill bar-final" style="width:{s100:.0f}%"></div></div>
      <span class="bar-val">{s100:.0f}</span>
    </div>
    <div class="bar-row">
      <span class="bar-label">Skills</span>
      <div class="bar-track"><div class="bar-fill bar-skill" style="width:{skill*100:.0f}%"></div></div>
      <span class="bar-val">{skill:.0%}</span>
    </div>
    <div class="bar-row">
      <span class="bar-label">Semantic</span>
      <div class="bar-track"><div class="bar-fill bar-embed" style="width:{embed*100:.0f}%"></div></div>
      <span class="bar-val">{embed:.0%}</span>
    </div>
    <div class="bar-row">
      <span class="bar-label">Ghost</span>
      <div class="bar-track"><div class="bar-fill bar-ghost" style="width:{ghost*100:.0f}%"></div></div>
      <span class="bar-val">{ghost*100:.0f}</span>
    </div>
  </div>
  <div class="card-reason">
    <span class="conf-pill {conf_cls}">{conf.title()}</span>&nbsp; {reason}
  </div>
  {skills_html}
</div>"""


def _podium_html(ranked: list) -> str:
    if len(ranked) < 1:
        return ""
    top3   = ranked[:3]
    medals = ["🥇", "🥈", "🥉"]
    clss   = ["pod-1", "pod-2", "pod-3"]
    cols   = ["#fbbf24", "#d1d5db", "#d97706"]
    order  = [1, 0, 2] if len(top3) == 3 else list(range(len(top3)))

    items = ""
    for idx in order:
        if idx >= len(top3):
            continue
        c    = top3[idx]
        cid  = c.get("candidate_id") or c.get("id") or f"#{idx+1}"
        t    = (c.get("title") or "")[:22]
        s100 = c.get("score_100", 0)
        gh   = c.get("ghost_score", 0)
        gem  = "⭐ " if c.get("hidden_gem") else ""
        items += (
            f'<div class="pod-col"><div class="pod-base {clss[idx]}">'
            f'<div style="font-size:38px">{medals[idx]}</div>'
            f'<div class="pod-id" style="color:{cols[idx]}">{gem}{cid}</div>'
            f'<div class="pod-title">{t}</div>'
            f'<div class="pod-score" style="color:{cols[idx]}">{s100:.0f}</div>'
            f'<div style="font-size:10px;color:rgba(255,255,255,.35)">/100</div>'
            f'<div class="pod-ghost">Ghost {gh*100:.0f}</div>'
            f"</div></div>"
        )
    return f'<div class="podium">{items}</div>'


def _leaderboard_html(ranked: list, stage_msg: str = "") -> str:
    if not ranked and not stage_msg:
        return '<div class="empty-state">No results yet.</div>'

    n_gems = sum(1 for c in ranked if c.get("hidden_gem"))
    n_dh   = sum(1 for c in ranked if c.get("dark_horse"))
    avg_v  = sum(c.get("volatility", 0) for c in ranked) / max(len(ranked), 1)
    top_g  = ranked[0].get("ghost_score", 0) if ranked else 0

    stats = (
        f'<div class="vk-stat-row">'
        f'<div class="vk-stat"><div class="vk-stat-n c-purple">{len(ranked)}</div><div class="vk-stat-l">Ranked</div></div>'
        f'<div class="vk-stat"><div class="vk-stat-n c-gold">{n_gems}</div><div class="vk-stat-l">Hidden Gems</div></div>'
        f'<div class="vk-stat"><div class="vk-stat-n c-blue">{n_dh}</div><div class="vk-stat-l">Dark Horses</div></div>'
        f'<div class="vk-stat"><div class="vk-stat-n c-purple">{top_g*100:.0f}</div><div class="vk-stat-l">Top Ghost</div></div>'
        f'<div class="vk-stat"><div class="vk-stat-n c-green">{avg_v:.2f}</div><div class="vk-stat-l">Avg Volatility</div></div>'
        f'</div>'
    ) if ranked else ""

    progress = f'<div class="vk-status">{stage_msg}</div>' if stage_msg else ""
    podium   = _podium_html(ranked) + '<hr style="border-color:rgba(255,255,255,.07);margin:20px 0">' if ranked else ""
    cards    = "".join(_card_html(i + 1, c) for i, c in enumerate(ranked))

    return stats + progress + podium + cards


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

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


def viveka_run(jd_text: str, rerank_n: int, pii_on: bool, audit_on: bool):
    """Generator: streams HTML leaderboard updates as each candidate is ranked."""
    if len(jd_text.strip()) < 30:
        yield _leaderboard_html([], ""), "", ""
        return

    os.environ["VIVEKA_PII_FIREWALL"] = "on" if pii_on else "off"
    os.environ["VIVEKA_AUDIT"]        = "on" if audit_on else "off"

    # Stage 1 ─────────────────────────────────────────────────────────────────
    yield _leaderboard_html([], "🔍 Stage 1 — Parsing job description with LLM…"), "", ""
    try:
        from jd_parser import parse_jd
        parsed_jd = parse_jd(jd_text)
    except Exception as exc:
        yield f'<div class="vk-error">Stage 1 failed: {exc}</div>', "", ""
        return

    req_s = ", ".join(parsed_jd.get("required_skills", [])[:6])
    jd_summary = (
        f'<div class="vk-jd-pills">'
        f'<span class="vk-jd-pill">Seniority: <b>{parsed_jd.get("seniority","?").title()}</b></span>'
        f'<span class="vk-jd-pill">Required: <b>{len(parsed_jd.get("required_skills",[]))}</b></span>'
        f'<span class="vk-jd-pill">Implied: <b>{len(parsed_jd.get("implied_skills",[]))}</b></span>'
        f'<span class="vk-jd-pill" style="color:rgba(255,255,255,.45)">{req_s}</span>'
        f"</div>"
    )

    # Stage 2 ─────────────────────────────────────────────────────────────────
    yield jd_summary + _leaderboard_html([], "⚡ Stage 2 — FAISS embedding recall…"), "", ""
    try:
        from recall import RecallEngine
        from agent  import load_profiles
        profiles = load_profiles(_data_dir)
        engine   = RecallEngine()
        engine.index_candidates(profiles)
        recalled = engine.recall(parsed_jd, top_k=min(200, len(profiles)))
    except Exception as exc:
        yield f'<div class="vk-error">Stage 2 failed: {exc}</div>', "", ""
        return

    # Stage 3 ─────────────────────────────────────────────────────────────────
    yield jd_summary + _leaderboard_html([], f"📊 Stage 3 — Scoring {len(recalled)} candidates…"), "", ""
    try:
        from scoring import score_candidates
        scored = score_candidates(recalled, parsed_jd, top_n=50)
    except Exception as exc:
        yield f'<div class="vk-error">Stage 3 failed: {exc}</div>', "", ""
        return

    # Stage 4 — streaming rerank ──────────────────────────────────────────────
    from rerank import rerank_stream
    from output import normalize_scores

    ranked = []
    total  = min(int(rerank_n), len(scored))
    t0     = time.time()

    for i, result in enumerate(rerank_stream(scored, parsed_jd, top_n=int(rerank_n)), 1):
        ranked.append(result)
        rs = sorted(ranked, key=lambda x: x["final_score"], reverse=True)
        rs = normalize_scores(rs)
        rs = _ghost(rs)

        elapsed = time.time() - t0
        pct     = i / total * 100
        prog    = (
            f'<div class="vk-progress"><div class="vk-progress-bar" style="width:{pct:.0f}%"></div></div>'
            f'<div style="font-size:11px;color:rgba(255,255,255,.4);margin-bottom:10px">'
            f"⚡ {i}/{total} ranked · {elapsed:.0f}s elapsed</div>"
        )
        yield jd_summary + prog + _leaderboard_html(rs), "", ""

    # Final
    rs   = _ghost(normalize_scores(sorted(ranked, key=lambda x: x["final_score"], reverse=True)))
    csv_ = _make_csv(rs)
    jstr = json.dumps(rs, indent=2, default=str)
    yield jd_summary + _leaderboard_html(rs), csv_, jstr


def _make_csv(ranked: list) -> str:
    import csv
    import io
    buf    = io.StringIO()
    fields = ["rank", "candidate_id", "title", "score_100", "final_score",
              "llm_score", "confidence", "hidden_gem", "ghost_score",
              "dark_horse", "volatility", "reason"]
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for i, c in enumerate(ranked, 1):
        row = dict(c)
        row["rank"]         = i
        row["candidate_id"] = c.get("candidate_id") or c.get("id", "")
        w.writerow(row)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Ghost Intel chart (returns Plotly figure)
# ---------------------------------------------------------------------------

def ghost_chart(json_str: str):
    if not json_str:
        return None
    try:
        import plotly.graph_objects as go
        ranked = json.loads(json_str)
        ids    = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(ranked)]
        actual = [c.get("final_score", 0) for c in ranked]
        ghost  = [c.get("ghost_score",  0) for c in ranked]
        colors = ["#f59e0b" if c.get("hidden_gem") else ("#a78bfa" if c.get("dark_horse") else "#60a5fa") for c in ranked]
        syms   = ["star"    if c.get("hidden_gem") else ("diamond"  if c.get("dark_horse") else "circle")  for c in ranked]

        fig = go.Figure()
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color="rgba(255,255,255,0.2)", dash="dot"))
        fig.add_annotation(x=0.7, y=0.85, text="Ghost > Actual = untapped potential",
                           showarrow=False, font=dict(color="rgba(255,255,255,0.35)", size=10))
        fig.add_trace(go.Scatter(
            x=actual, y=ghost, mode="markers+text",
            text=ids, textposition="top center", textfont=dict(size=9),
            marker=dict(size=14, color=colors, symbol=syms,
                        line=dict(width=1, color="rgba(255,255,255,.25)")),
            hovertemplate="<b>%{text}</b><br>Actual: %{x:.3f}<br>Ghost: %{y:.3f}<extra></extra>",
            showlegend=False,
        ))
        fig.update_layout(
            title="Ghost Score vs Actual Score",
            xaxis=dict(title="Actual Final Score", range=[0, 1.05]),
            yaxis=dict(title="Ghost Score (Potential)", range=[0, 1.05]),
            template="plotly_dark", height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,7,15,0.6)",
        )
        return fig
    except Exception:
        return None


def volatility_chart(json_str: str):
    if not json_str:
        return None
    try:
        import plotly.graph_objects as go
        ranked = json.loads(json_str)
        ids    = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(ranked)]
        volats = [c.get("volatility", 0) for c in ranked]
        pairs  = sorted(zip(ids, volats), key=lambda x: x[1], reverse=True)[:12]
        v_ids  = [p[0] for p in pairs]
        v_vals = [p[1] for p in pairs]
        colors = ["#ef4444" if v > 0.3 else ("#f59e0b" if v > 0.1 else "#10b981") for v in v_vals]

        fig = go.Figure(go.Bar(
            x=v_ids, y=v_vals, marker_color=colors,
            text=[f"{v:.2f}" for v in v_vals], textposition="outside",
        ))
        fig.update_layout(
            title="Ranking Volatility Index",
            yaxis=dict(title="Volatility", range=[0, max(v_vals or [0.1]) * 1.35]),
            xaxis_tickangle=-30, template="plotly_dark", height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,7,15,0.6)",
        )
        return fig
    except Exception:
        return None


def score_breakdown_chart(json_str: str):
    if not json_str:
        return None
    try:
        import plotly.graph_objects as go
        ranked = json.loads(json_str)
        top    = ranked[:10]
        ids    = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(top)]
        sigs   = {
            "Embedding":  [c.get("embedding_score",  0) for c in top],
            "Skill":      [c.get("skill_score",       0) for c in top],
            "Seniority":  [c.get("seniority_score",   0) for c in top],
            "Activity":   [c.get("activity_score",    0) for c in top],
        }
        colors = {"Embedding": "#4e9af1", "Skill": "#7ce38b", "Seniority": "#f0ad4e", "Activity": "#e07db3"}
        fig = go.Figure()
        for sig, vals in sigs.items():
            fig.add_trace(go.Bar(name=sig, x=ids, y=vals, marker_color=colors[sig]))
        fig.add_trace(go.Scatter(
            x=ids, y=[c.get("composite_score", 0) for c in top],
            name="Composite", mode="lines+markers",
            line=dict(color="#fff", width=2, dash="dot"),
        ))
        fig.update_layout(
            barmode="stack", title="Signal Contribution — Top 10",
            yaxis=dict(range=[0, 1.2], title="Score"),
            xaxis_tickangle=-30, template="plotly_dark", height=380,
            legend=dict(orientation="h", y=-0.28),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,7,15,0.6)",
        )
        return fig
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

HERO = """
<div class="vk-hero">
  <div class="vk-logo">🔍 VIVEKA</div>
  <div class="vk-tagline">Discerning True Talent from Noise</div>
  <div class="vk-sub">Ghost Intelligence · What-If Simulator · Citation Grounding · PII Firewall</div>
</div>
"""

if _provider == "gemini":
    PROVIDER_BADGE = '<div class="provider-badge">☁️ Cloud · Gemini API</div>'
elif _provider == "huggingface":
    PROVIDER_BADGE = '<div class="provider-badge">🤗 Cloud · HF Inference API (free)</div>'
else:
    PROVIDER_BADGE = '<div class="provider-badge">🖥️ Local · Ollama</div>'

EMPTY = '<div class="empty-state">Paste a job description and click <b>▶ Run VIVEKA</b> to rank candidates.</div>'


with gr.Blocks(
    css=CSS,
    title="VIVEKA — Intelligent Candidate Ranking",
    theme=gr.themes.Base(
        primary_hue="purple",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
) as demo:

    gr.HTML(HERO)

    # ── State ────────────────────────────────────────────────────────────────
    json_state = gr.State("")

    with gr.Row(equal_height=False):

        # ── Left panel — input ────────────────────────────────────────────
        with gr.Column(scale=1, min_width=330):
            gr.Markdown("### 📋 Job Description")
            jd_input = gr.Textbox(
                value=_SAMPLE_JD,
                lines=13,
                placeholder="Paste the full job description here…",
                show_label=False,
            )

            with gr.Accordion("⚙️ Settings", open=True):
                rerank_n = gr.Slider(
                    minimum=3, maximum=50, value=8, step=1,
                    label="Candidates to rerank (LLM calls)",
                    info="Lower = faster. Raise to 50 for final submission.",
                )
                pii_toggle = gr.Checkbox(
                    value=True,
                    label="🔒 PII Firewall — strip identity before scoring",
                )
                audit_toggle = gr.Checkbox(
                    value=True,
                    label="📝 Audit Trail — log to audit.jsonl",
                )

            run_btn = gr.Button("▶  Run VIVEKA", variant="primary", size="lg")
            gr.HTML(PROVIDER_BADGE)

            gr.Markdown("---")
            gr.Markdown("### ⬇️ Download Results")
            csv_dl  = gr.File(label="ranked_output.csv",  visible=False)
            json_dl = gr.File(label="ranked_output.json", visible=False)

        # ── Right panel — results ─────────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Tabs():

                with gr.Tab("🏆 Rankings"):
                    results_html = gr.HTML(value=EMPTY)

                with gr.Tab("👻 Ghost Intel"):
                    ghost_plot  = gr.Plot(label="Ghost vs Actual")
                    volat_plot  = gr.Plot(label="Volatility Index")
                    gr.Markdown(
                        "**Legend:** ⭐ Star = Hidden Gem · 💎 Diamond = Dark Horse · 🔵 Circle = Standard  \n"
                        "**Volatility:** 🔴 > 0.30 unstable · 🟡 0.10–0.30 moderate · 🟢 < 0.10 stable"
                    )

                with gr.Tab("📊 Score Breakdown"):
                    breakdown_plot = gr.Plot(label="Signal Contribution")

                with gr.Tab("🔮 What-If Simulator"):
                    gr.Markdown(
                        "Select a candidate and check missing skills to see the hypothetical rank impact. "
                        "Run the pipeline first."
                    )
                    wi_cand   = gr.Dropdown(label="Candidate", choices=[], interactive=True)
                    wi_skills = gr.CheckboxGroup(label="Missing skills to add", choices=[], interactive=True)
                    wi_result = gr.HTML()
                    wi_btn    = gr.Button("Simulate", variant="secondary")

    # ── Internal CSV string for download ─────────────────────────────────────
    csv_str_state  = gr.State("")
    jd_state       = gr.State({})
    ranked_state   = gr.State([])

    # ── Run pipeline ──────────────────────────────────────────────────────────
    def _run(jd, n, pii, audit):
        csv_out = ""
        jstr    = ""
        for html, csv_s, j in viveka_run(jd, n, pii, audit):
            csv_out = csv_s or csv_out
            jstr    = j    or jstr
            yield html, csv_out, jstr

    run_event = run_btn.click(
        fn=_run,
        inputs=[jd_input, rerank_n, pii_toggle, audit_toggle],
        outputs=[results_html, csv_str_state, json_state],
    )

    # ── After pipeline: update charts + what-if dropdowns ────────────────────
    def _post_run(jstr: str, jd_text: str):
        if not jstr:
            return None, None, None, gr.update(choices=[]), gr.update(choices=[]), []

        ranked = json.loads(jstr)

        # Parse JD for what-if
        ids = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(ranked)]

        return (
            ghost_chart(jstr),
            volatility_chart(jstr),
            score_breakdown_chart(jstr),
            gr.update(choices=ids, value=ids[0] if ids else None),
            gr.update(choices=[]),
            ranked,
        )

    run_btn.click(
        fn=_post_run,
        inputs=[json_state, jd_input],
        outputs=[ghost_plot, volat_plot, breakdown_plot, wi_cand, wi_skills, ranked_state],
    )

    # ── What-If: update skill checkboxes when candidate changes ───────────────
    def _wi_candidate_change(sel_id: str, ranked: list):
        if not sel_id or not ranked:
            return gr.update(choices=[])
        ids = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(ranked)]
        if sel_id not in ids:
            return gr.update(choices=[])
        c  = ranked[ids.index(sel_id)]
        ev = c.get("skill_evidence", {})
        missing = ev.get("required_missing", []) + ev.get("implied_missing", [])
        return gr.update(choices=missing, value=[])

    wi_cand.change(fn=_wi_candidate_change, inputs=[wi_cand, ranked_state], outputs=[wi_skills])

    # ── What-If: simulate ─────────────────────────────────────────────────────
    def _wi_simulate(sel_id: str, added_skills: list, ranked: list, jd_text: str):
        if not sel_id or not ranked or not added_skills:
            return "<div style='color:rgba(255,255,255,.4);font-size:13px'>Select a candidate and check skills to add.</div>"

        ids = [c.get("candidate_id") or c.get("id") or f"#{i+1}" for i, c in enumerate(ranked)]
        if sel_id not in ids:
            return ""
        c       = ranked[ids.index(sel_id)]
        old_rank = ids.index(sel_id) + 1
        ev       = c.get("skill_evidence", {})

        try:
            from jd_parser import parse_jd
            pjd = parse_jd(jd_text)
        except Exception:
            pjd = {}

        required = pjd.get("required_skills", [])
        implied  = pjd.get("implied_skills",  [])

        req_miss  = ev.get("required_missing", [])
        impl_miss = ev.get("implied_missing",  [])

        added_req  = [s for s in added_skills if s in req_miss]
        added_impl = [s for s in added_skills if s in impl_miss]

        old_req_h  = len(ev.get("required_matched", []))
        old_impl_h = len(ev.get("implied_matched",  []))
        new_req_h  = old_req_h  + len(added_req)
        new_impl_h = old_impl_h + len(added_impl)

        max_s      = len(required) * 1.0 + len(implied) * 0.5 or 1.0
        old_raw    = (old_req_h * 1.0 + old_impl_h * 0.5) / max_s
        new_raw    = (new_req_h * 1.0 + new_impl_h * 0.5) / max_s
        stuff      = c.get("stuffing", {}).get("stuffing_ratio", 0)
        penalty    = 0.3
        old_skill  = old_raw * (1 - stuff * penalty)
        new_skill  = new_raw * (1 - stuff * penalty)
        delta_s    = new_skill - old_skill
        w_skill    = 0.4
        comp_delta = delta_s * w_skill

        hypo = [(c2.get("composite_score", 0), ids[j])
                for j, c2 in enumerate(ranked) if ids[j] != sel_id]
        hypo.append((c.get("composite_score", 0) + comp_delta, sel_id))
        hypo.sort(reverse=True)
        new_rank    = next(j for j, (_, cid) in enumerate(hypo, 1) if cid == sel_id)
        rank_delta  = old_rank - new_rank
        arrow       = f"↑ {rank_delta} places" if rank_delta > 0 else ("No change" if rank_delta == 0 else f"↓ {abs(rank_delta)} places")
        arrow_color = "#34d399" if rank_delta > 0 else ("#fbbf24" if rank_delta == 0 else "#f87171")

        return f"""
<div style="background:rgba(124,58,237,.1);border:1px solid rgba(124,58,237,.3);border-radius:14px;padding:18px 22px;">
  <div style="font-size:13px;color:rgba(255,255,255,.6);margin-bottom:12px">
    Adding <b style="color:#a78bfa">{", ".join(added_skills)}</b> to <b>{sel_id}</b>:
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">
    <div style="text-align:center">
      <div style="font-size:11px;color:rgba(255,255,255,.4)">Skill Score</div>
      <div style="font-size:22px;font-weight:800;color:#34d399">{old_skill:.0%}→{new_skill:.0%}</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:11px;color:rgba(255,255,255,.4)">Composite Δ</div>
      <div style="font-size:22px;font-weight:800;color:#60a5fa">+{comp_delta:.3f}</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:11px;color:rgba(255,255,255,.4)">Current Rank</div>
      <div style="font-size:22px;font-weight:800;color:#e2e8f0">#{old_rank}</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:11px;color:rgba(255,255,255,.4)">New Rank</div>
      <div style="font-size:22px;font-weight:800;color:{arrow_color}">#{new_rank}</div>
      <div style="font-size:11px;color:{arrow_color}">{arrow}</div>
    </div>
  </div>
</div>"""

    wi_btn.click(
        fn=_wi_simulate,
        inputs=[wi_cand, wi_skills, ranked_state, jd_input],
        outputs=[wi_result],
    )

demo.queue()

if __name__ == "__main__":
    demo.launch()
