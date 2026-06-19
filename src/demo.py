"""
demo.py — Streamlit Demo UI

Paste a job description → click Run → see the ranked shortlist
with scores, LLM reasons, and hidden gem flags.

Run with:  streamlit run src/demo.py
"""

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from agent import load_profiles
from output import print_summary

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
    st.header("Settings")

    model_name = st.text_input(
        "Ollama model",
        value=os.getenv("MANTHAN_MODEL", "llama3.2"),
        help="Must be pulled with `ollama pull <model>`",
    )
    os.environ["MANTHAN_MODEL"] = model_name

    rerank_n = st.slider(
        "Candidates to rerank (LLM calls)",
        min_value=3, max_value=50, value=10,
        help="Lower = faster. Raise to 50 for final run.",
    )

    st.divider()
    st.subheader("Weights")
    w_embed    = st.slider("Embedding weight",    0.0, 1.0, 0.40, 0.05)
    w_skill    = st.slider("Skill match weight",  0.0, 1.0, 0.40, 0.05)
    w_activity = st.slider("Activity weight",     0.0, 1.0, 0.20, 0.05)
    os.environ["MANTHAN_W_EMBED"]    = str(w_embed)
    os.environ["MANTHAN_W_SKILL"]    = str(w_skill)
    os.environ["MANTHAN_W_ACTIVITY"] = str(w_activity)

    st.divider()
    st.markdown("**How to switch model:**")
    st.code("ollama pull mistral:7b\n# then set model name above")

# ---------------------------------------------------------------------------
# Main — job description input
# ---------------------------------------------------------------------------
st.subheader("Job Description")

sample_jd = """Senior Data Engineer — Platform Team

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
    value=sample_jd,
    height=220,
)

# ---------------------------------------------------------------------------
# Candidate data source
# ---------------------------------------------------------------------------
data_dir    = Path(__file__).parent.parent / "data"
json_path   = data_dir / "profiles.json"
csv_path    = data_dir / "profiles.csv"

if json_path.exists():
    data_source = f"profiles.json ({len(json.loads(json_path.read_text()))} candidates)"
elif csv_path.exists():
    data_source = f"profiles.csv"
else:
    data_source = "Built-in demo profiles (8 candidates)"

st.caption(f"Candidate data: {data_source}")

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------
run_btn = st.button("▶  Run MANTHAN", type="primary", use_container_width=True)

if run_btn:
    if not jd_text.strip():
        st.error("Please enter a job description.")
        st.stop()

    profiles = load_profiles(data_dir)

    # Cache the embedding index — rebuilding costs 30-60s on a large dataset.
    # Keyed on the data directory path; invalidated automatically if profiles change.
    @st.cache_resource(show_spinner="Loading embedding model and indexing profiles…")
    def _get_index(data_dir_key: str):
        from recall import RecallEngine
        from agent import load_profiles as _lp
        _profiles = _lp(Path(data_dir_key))
        eng = RecallEngine()
        eng.index_candidates(_profiles)
        return eng, _profiles

    with st.status("Running MANTHAN pipeline…", expanded=True) as status:
        t0 = time.time()

        st.write("Stage 1 — Parsing job description with local LLM…")
        try:
            from jd_parser import parse_jd
            parsed_jd = parse_jd(jd_text)
        except Exception as e:
            st.error(f"Stage 1 failed: {e}")
            st.stop()

        st.write(f"Stage 2 — FAISS recall (model cached after first run)…")
        engine, profiles = _get_index(str(data_dir))
        recalled = engine.recall(parsed_jd, top_k=min(200, len(profiles)))

        st.write("Stage 3 — Multi-signal scoring…")
        from scoring import score_candidates
        scored = score_candidates(recalled, parsed_jd, top_n=50)

        st.write(f"Stage 4 — LLM rerank (top {rerank_n})…")
        from rerank import rerank_candidates
        ranked = rerank_candidates(scored, parsed_jd, top_n=rerank_n)

        elapsed = time.time() - t0
        status.update(label=f"Done in {elapsed:.1f}s", state="complete")

    # ── Results ───────────────────────────────────────────────────────────────

    st.subheader("Parsed Job Requirements")
    col1, col2, col3 = st.columns(3)
    col1.metric("Seniority", parsed_jd.get("seniority", "—").title())
    col2.metric("Required skills", len(parsed_jd.get("required_skills", [])))
    col3.metric("Implied skills",  len(parsed_jd.get("implied_skills",  [])))

    with st.expander("See full parsed JD"):
        st.json(parsed_jd)

    # Hidden gems callout
    gems = [c for c in ranked if c.get("hidden_gem")]
    if gems:
        st.success(f"★ {len(gems)} Hidden Gem(s) found — strong candidates with unassuming profiles")

    # Ranked table
    st.subheader(f"Ranked Shortlist — Top {len(ranked)}")

    for i, c in enumerate(ranked, 1):
        cid    = c.get("candidate_id") or c.get("id") or f"#{i}"
        title  = c.get("title", "")
        score  = c.get("final_score", 0.0)
        llm    = c.get("llm_score", "—")
        conf   = c.get("confidence", "—")
        reason = c.get("reason", "No reason provided.")
        is_gem = c.get("hidden_gem", False)

        gem_badge = "  ★ Hidden Gem" if is_gem else ""
        label = f"**#{i} — {cid}** | {title}{gem_badge}"

        with st.expander(label, expanded=(i <= 3)):
            cols = st.columns(4)
            cols[0].metric("Final Score", f"{score:.3f}")
            cols[1].metric("LLM Score",   f"{llm}/10")
            cols[2].metric("Confidence",  conf.title() if isinstance(conf, str) else conf)
            cols[3].metric("Skill Match", f"{c.get('skill_score', 0):.2%}")

            st.markdown(f"**Reason:** {reason}")

            with st.expander("Full profile"):
                display = {k: v for k, v in c.items()
                           if not k.startswith("_") and k not in
                           ("reason", "final_score", "llm_score", "confidence",
                            "composite_score", "skill_score", "activity_score",
                            "embedding_score", "hidden_gem")}
                st.json(display)

    # Download
    st.divider()
    import csv, io
    buf = io.StringIO()
    fields = ["rank", "candidate_id", "title", "final_score", "llm_score",
              "confidence", "hidden_gem", "reason", "skill_score", "composite_score"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for i, c in enumerate(ranked, 1):
        row = dict(c)
        row["rank"]         = i
        row["candidate_id"] = c.get("candidate_id") or c.get("id", "")
        writer.writerow(row)

    st.download_button(
        "⬇  Download ranked_output.csv",
        data=buf.getvalue(),
        file_name="ranked_output.csv",
        mime="text/csv",
        use_container_width=True,
    )
