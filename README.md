# MANTHAN — Churning the Ocean of Talent

> *Manthan* (Sanskrit: मंथन) — the great churning. In mythology, gods and demons
> churned the cosmic ocean to surface hidden treasures. We churn the talent pool
> to surface the right person for the right role.

Built for the **India Runs Data & AI Challenge**.

---

## The Vision

Most hiring tools match keywords. They miss the candidate who has never used
the exact tool but will master it in a week. They surface the loudest résumé,
not the best fit.

MANTHAN is different. It reads a job description the way a great recruiter does —
understanding what the role *really* needs, not just what it says. It ranks
candidates by evidence, flags hidden gems, and explains every choice in plain words.

The system is built on four pillars. One is fully working today. The other three
are the roadmap — described honestly as future work, never faked.

---

## Runs Fully Offline — Free, Private, No API Key

MANTHAN uses a **local language model** (Ollama + Llama 3.2). It needs no paid API
and no internet connection. Candidate data never leaves the machine.

This is not a workaround. It is a design choice. A hiring tool that sends résumés
to a third-party cloud raises real privacy concerns. MANTHAN avoids that entirely.

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

The fully built, running core of MANTHAN. An agent that reads a job, understands
what it truly needs, and produces a ranked shortlist with honest explanations.
Runs entirely on a local model — no cloud required.

---

## How Pillar 4 Works: The 5-Stage Engine

```
Job Description (raw text)
        │
        ▼
┌──────────────────────────────┐
│  Stage 1 · JD Intelligence   │  Local Llama reads the JD → required skills,
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
│  Stage 4 · Honest Rerank     │  Local Llama reads each profile + JD.
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

A candidate is flagged as a hidden gem when their composite score lands in the
top tier but their profile is sparse or their title doesn't signal seniority.
These are the candidates a keyword filter would have dropped. MANTHAN surfaces them.

---

## Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| Embeddings | `sentence-transformers` | Free, local |
| Fast search | `FAISS` | Free, local |
| Scoring/fusion | `scikit-learn` | Free, local |
| Language model | `Ollama` + Llama 3.2 | Free, local |
| LLM framework | `LangChain` | Free |
| Demo UI | `Streamlit` | Free |
| Optional fallback | Gemini free tier | Free (cloud) |

---

## Project Structure

```
manthan/
  data/            ← dataset (not committed; add your files here)
  src/
    llm.py         ← model provider (swap Ollama ↔ Gemini here only)
    agent.py       ← Pillar 4 orchestrator
    jd_parser.py   ← Stage 1: JD Intelligence
    recall.py      ← Stage 2: Fast Recall
    scoring.py     ← Stage 3: Multi-signal Scoring
    rerank.py      ← Stage 4: Honest Rerank
    output.py      ← Stage 5: Output Writer
    demo.py        ← Streamlit demo UI
  README.md
  requirements.txt
  CLAUDE.md
```

---

## Setup

```bash
# 1. Install Ollama (one-time)
#    https://ollama.com — download and install for your OS

# 2. Pull the model (one-time, ~2 GB)
ollama pull llama3.2

# 3. Start the local model server (keep this running)
ollama serve

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. (Optional) change the model without editing code
$env:MANTHAN_MODEL = "llama3.1"   # PowerShell
export MANTHAN_MODEL=llama3.1     # bash
```

---

## What We Are NOT Doing (and Why)

We are not faking pillars 1–3. We describe them as the honest roadmap they are.

We are not inventing performance numbers. Every claim in this system is backed by
observable outputs from the running pipeline.

We are not sending candidate data to the cloud. The local model is the feature,
not the compromise.

We win on quality and honesty — not on a slide deck of unbuilt features.

---

## Honesty Statement

> Pillars 1, 2, and 3 are vision. They are described to show where MANTHAN is
> going, not where it is today. Pillar 4 is the working system. Every shortlist
> it produces is grounded in real profile text. If the evidence is thin, it says so.
