---
title: VIVEKA
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.19.0"
app_file: app.py
pinned: false
---

# VIVEKA — Discerning True Talent from Noise

> *Viveka* (Sanskrit: विवेक) — discernment. The clarity to tell true from false,
> signal from noise. In classical thought, viveka is the faculty that sees past
> surface appearances to perceive what is real.
>
> We apply that faculty to hiring: discerning a candidate's genuine capability
> from keyword noise and surface-level résumé claims.

Built for the **India Runs Data & AI Challenge** — Track 1: AI-Powered Candidate Ranking.

---

## The Vision

Most hiring tools match keywords. They surface the résumé that mentions
"Apache Spark" six times — not the engineer who built production pipelines with it.
They reward noise over signal.

VIVEKA is different. It reads a job description the way a seasoned recruiter does —
understanding what the role *truly* needs, not just what the text says. It scores
candidates on genuine evidence, flags hidden gems whose titles undersell their
ability, and explains every ranking decision in plain words.

The system is built on four pillars. One is fully working today. The other three
are the honest roadmap — described as future work, never faked.

---

## Runs Fully Offline — Free, Private, No API Key

VIVEKA uses a **local language model** (Ollama + Llama 3.2). It needs no paid API
and no internet connection. Candidate data never leaves the machine.

This is not a workaround. It is a design choice. A hiring tool that sends résumés
to a third-party cloud raises real privacy concerns. VIVEKA avoids that entirely.

**Cloud mode** (Gemini free tier) is available for demos on Hugging Face Spaces.

---

## The Four Pillars

### Pillar 1 — Talent Knowledge Graph `[ROADMAP]`

A graph database (Neo4j or similar) that maps skills, roles, industries, and
transitions. It knows that "Spark" implies distributed systems, that a "senior
analyst at a fintech" likely knows SQL and Python, and that certain career paths
are stepping stones to others.

**Why it matters:** turns isolated résumé facts into a connected map of talent.
**Status:** design complete. Build follows dataset availability.

---

### Pillar 2 — Trajectory Modeling `[ROADMAP]`

A sequence model trained on career histories. It predicts where a candidate is
heading, not just where they have been. A junior engineer with an accelerating
skill curve ranks higher than a senior who has stagnated.

**Why it matters:** hiring is a bet on the future, not a reward for the past.
**Status:** architecture defined. Requires longitudinal career data to train.

---

### Pillar 3 — Candidate Digital Twin `[ROADMAP]`

A profile enrichment layer that infers missing signals — inferred skills from job
titles, likely tools from industries, probable soft skills from tenure and team size.
Not invented facts; probabilistic completion of sparse profiles.

**Why it matters:** most candidate data is sparse. Imputation without hallucination
unlocks the long tail of overlooked applicants.
**Status:** approach defined. Tied to Pillar 1 graph for inference paths.

---

### Pillar 4 — Agentic Recruiter + Ranking `[WORKING NOW]`

The fully built, running core of VIVEKA. An agent that reads a job, understands
what it truly needs, and produces a ranked shortlist with honest explanations.
Runs entirely on a local model — no cloud required.

---

## How Pillar 4 Works: The 5-Stage Engine

```
Job Description (raw text)
        │
        ▼
┌──────────────────────────────┐
│  Stage 1 · JD Intelligence   │  Local LLM reads the JD → required skills,
│  (jd_parser.py)              │  implied skills, seniority, latent needs
└─────────────┬────────────────┘
              │ structured JSON
              ▼
┌──────────────────────────────┐
│  Stage 2 · Fast Recall       │  Embed JD + all profiles with
│  (recall.py)                 │  sentence-transformers. FAISS retrieves
└─────────────┬────────────────┘  top ~200 candidates in milliseconds.
              │ top-200 candidates
              ▼
┌──────────────────────────────┐
│  Stage 3 · Multi-signal      │  Blend three signals:
│  Scoring (scoring.py)        │  · semantic similarity (embedding dot product)
└─────────────┬────────────────┘  · skill overlap (weighted by importance)
              │                   · activity / behavior signals
              │ ranked top-50
              ▼
┌──────────────────────────────┐
│  Stage 4 · Honest Rerank     │  Local LLM reads each profile + JD.
│  (rerank.py)                 │  Scores fit. Writes a reason using REAL
└─────────────┬────────────────┘  text. If proof is weak, it says so.
              │ scored + explained
              ▼
┌──────────────────────────────┐
│  Stage 5 · Output            │  Writes the ranked list in the organizers'
│  (output.py)                 │  format: rank, ID, score, reason,
└──────────────────────────────┘  and "hidden gem" flag.
```

### The "Hidden Gem" Flag

A candidate is flagged as a hidden gem when their composite score (multi-signal)
lands in the top tier but their raw embedding rank placed them lower. These are
the candidates a keyword filter would have dropped — VIVEKA's discernment surfaces them.

---

## Trust Features

### PII Firewall
All identity fields (name, email, phone, gender, age, location, nationality,
photo, address) are stripped from every profile **before** scoring or LLM rerank.
Enabled by default. Ranking is based on skills and evidence alone.

Toggle: `VIVEKA_PII_FIREWALL=off` to disable (not recommended).

### Citation Grounding
The LLM must provide verbatim snippets from the profile as evidence for its score.
A deterministic verifier checks each snippet against the actual profile text.
If a citation is not found, confidence is automatically downgraded and the
snippet is flagged as `evidence_unsupported`.

### Keyword-Stuffing Detector
Skills listed in structured fields but never mentioned in narrative prose are
flagged as `claimed_unsupported`. A penalty (default 30%) is applied to
`skill_score` — honest discounting, not disqualification.

### Full Audit Trail
Every pipeline run appends to `data/audit.jsonl`. Each record contains the
run ID (SHA-256 hash of JD + model + weights), all scores, evidence, and the
model used. The same inputs always produce the same run ID — reproducibility
is provable.

---

## Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| Embeddings | `sentence-transformers` | Free, local |
| Fast search | `FAISS` (`faiss-cpu`) | Free, local |
| Scoring/fusion | `scikit-learn` | Free, local |
| Language model | `Ollama` + Llama 3.2 | Free, local |
| LLM framework | `LangChain` | Free |
| Cloud fallback | `langchain-google-genai` (Gemini) | Free tier |
| Demo UI | `Streamlit` | Free |

---

## Project Structure

```
viveka/
  app.py              ← Streamlit entry-point (HF Spaces + local)
  data/
    job_description.txt      ← sample JD
    sample_candidates.json   ← 15 demo profiles (no real PII)
    ranked_output.json       ← submission: full scored output
    ranked_output.csv        ← submission: CSV shortlist
  src/
    llm.py            ← model provider (swap Ollama ↔ Gemini here only)
    agent.py          ← Pillar 4 orchestrator
    data_loader.py    ← flexible .json/.csv loader with column normalisation
    jd_parser.py      ← Stage 1: JD Intelligence
    recall.py         ← Stage 2: Fast Recall
    scoring.py        ← Stage 3: Multi-signal Scoring + stuffing detector
    rerank.py         ← Stage 4: Honest Rerank + citation grounding
    output.py         ← Stage 5: Output Writer + validate_output()
    pii.py            ← PII Firewall
    audit.py          ← Audit trail
    config.py         ← All constants and env-var defaults
    skills.py         ← Skill synonyms (Spark = PySpark = Apache Spark)
    demo.py           ← Full Streamlit demo UI
  tests/              ← 158 tests, all passing
  requirements.txt
  CLAUDE.md
```

---

## How to Run

### Local — Ollama (free, offline, private)

```bash
# 1. Install Ollama: https://ollama.com

# 2. Pull the model (~4 GB, one-time)
ollama pull llama3.2

# 3. Keep Ollama server running
ollama serve

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Run the pipeline (CLI)
python src/agent.py

# 6. Or run the Streamlit demo
streamlit run app.py
```

You can swap to a smaller/faster model without editing code:

```bash
# PowerShell
$env:VIVEKA_MODEL = "phi3:mini"
# bash
export VIVEKA_MODEL=phi3:mini
```

### Cloud Demo — Gemini (Hugging Face Spaces)

```bash
# Set your free Gemini key (https://aistudio.google.com)
$env:LLM_PROVIDER = "gemini"
$env:GOOGLE_API_KEY = "AIza..."
streamlit run app.py
```

### With Your Own Dataset

Drop your candidate file into `data/` as `profiles.json` or `profiles.csv`.
The loader auto-detects column names and normalises variants
(`headline`→`title`, `about`→`summary`, `tech_skills`→`skills`, etc.).

Then run: `python src/agent.py`

---

## Results

The pipeline was run on `data/sample_candidates.json` (15 realistic profiles
for a Senior Data Engineer role). Full output:

- `data/ranked_output.json` — complete scored shortlist with evidence, stuffing
  analysis, counterfactual explainers, and audit metadata
- `data/ranked_output.csv`  — CSV shortlist in submission format

**Validation:** all 4 checks PASS — sequential ranks, no null scores,
sorted correctly, all reasons non-empty.

Hidden gems detected: candidates with modest titles (Analytics Associate,
Data Support Specialist) who demonstrated strong pipeline engineering in their
experience text were surfaced above higher-titled candidates with weaker evidence.
This is viveka at work: the discernment to see past title to genuine capability.

---

## What We Are NOT Doing (and Why)

We are not faking pillars 1–3. We describe them as the honest roadmap they are.

We are not inventing performance numbers. Every claim is backed by observable
outputs from the running pipeline.

We are not sending candidate data to the cloud. The local model is the feature,
not the compromise.

We win on quality and honesty — not on a slide deck of unbuilt features.

---

## Honesty Statement

> Pillars 1, 2, and 3 are vision. They are described to show where VIVEKA is
> going, not where it is today. Pillar 4 is the working system. Every shortlist
> it produces is grounded in real profile text. If the evidence is thin, it says so.
>
> VIVEKA — the discernment to separate signal from noise — is not just the name.
> It is the design principle.
