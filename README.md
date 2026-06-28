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

# VIVEKA — Find the Best Candidate. Every Time.

> **Viveka** (Sanskrit: विवेक) means *discernment* — the wisdom to tell real from fake,
> signal from noise, genuine skill from a well-written résumé.
>
> That is exactly what this system does for hiring.

**🚀 Live Demo:** [https://huggingface.co/spaces/mohareddy1423/VIVEKA](https://huggingface.co/spaces/mohareddy1423/VIVEKA)

**🐙 GitHub:** [https://github.com/Mohanreddy-lab/VIVEKA](https://github.com/Mohanreddy-lab/VIVEKA)

Built for the **India Runs Data & AI Challenge — Track 1: AI-Powered Candidate Ranking**

---

## What Problem Does This Solve?

Imagine you are hiring a Data Engineer. You get 500 applications.

A normal hiring tool will rank the person who wrote "Apache Spark" 12 times at the top.
But is that person actually good? Maybe not.

The best candidate might be someone who:
- Has a modest job title like "Data Analyst"
- Never keyword-stuffed their résumé
- But built real Spark pipelines that processed millions of records

A normal tool misses them. **VIVEKA finds them.**

---

## What Makes VIVEKA Different?

| Normal Tool | VIVEKA |
|---|---|
| Counts keyword matches | Understands what the job *really* needs |
| Rewards résumé stuffing | Penalizes fake skill claims |
| Ranks by title/years | Ranks by actual evidence in the text |
| Black box — no explanation | Explains every rank in plain words |
| Ignores hidden talent | Flags "Hidden Gems" — overlooked but strong |
| Sends data to the cloud | Runs fully offline — your data stays private |

---

## How It Works — 5 Stages Explained Simply

```
You paste a Job Description
           ↓
   Stage 1: What does this job really need?
           ↓
   Stage 2: Find the closest candidates fast (top 200)
           ↓
   Stage 3: Score them on 3 signals
           ↓
   Stage 4: AI reads each profile + writes honest explanations
           ↓
   Stage 5: Final ranked list with scores, reasons, download
```

---

### Stage 1 — JD Intelligence (`jd_parser.py`)

The AI reads your job description like a senior recruiter would.

It extracts:
- **Required skills** — things you must have (e.g., Python, Spark)
- **Implied skills** — things not written but obviously needed (e.g., if you need "Spark", you probably need distributed computing knowledge)
- **Seniority level** — junior / mid / senior / lead, detected from wording
- **Latent needs** — e.g., "cross-functional collaboration" buried in a sentence about reporting

Output: clean structured JSON used by every stage after this.

---

### Stage 2 — Fast Recall (`recall.py`)

You might have 1,000 candidates. Checking all 1,000 with an AI would take hours and cost a lot.

Instead, VIVEKA:
1. Converts the job description and all profiles into **embeddings** (numbers that capture meaning)
2. Uses **FAISS** (a lightning-fast search index by Meta) to find the top 200 candidates in milliseconds
3. These 200 go to Stage 3 — the rest are filtered out quickly

On Hugging Face Spaces: embeddings are generated using the HF Inference API (no local GPU needed).
Locally: uses `sentence-transformers` if installed, same API fallback otherwise.

---

### Stage 3 — Multi-Signal Scoring (`scoring.py`)

For each of the top 200, VIVEKA blends **3 signals** into one composite score:

**Signal 1 — Semantic Match (embedding similarity)**
How close is the candidate's profile to the job description in *meaning*, not just words?
A candidate who says "built data pipelines on AWS" scores high for a job that says "cloud data infrastructure" — even though the exact words differ.

**Signal 2 — Skill Overlap (weighted)**
Are the required skills present? Are the *important* skills present? Each required skill has a weight. Missing a core skill hurts more than missing a nice-to-have.

Special protections:
- **Keyword Stuffing Detector** — if a skill appears in the structured "skills" field but NEVER appears in the actual experience text, it is flagged as fake and the score is reduced 30%
- **Skill Synonym Matching** — "Spark", "PySpark", "Apache Spark" are all the same thing. VIVEKA knows this.

**Signal 3 — Activity & Behavior Signals**
If the data includes GitHub activity, publication count, project count, or tenure signals — these feed into the score. Candidates who *do* things rank above candidates who only *claim* things.

**Final composite = weighted blend of all 3 signals**

---

### Stage 4 — Honest Rerank (`rerank.py`)

The top 50 candidates go to the AI for a deep read.

For each candidate, the AI:
1. Reads the actual profile text (after PII is stripped)
2. Scores the fit from 1–10
3. Writes a reason in plain English using **real quotes** from the profile
4. Sets confidence: `high` / `medium` / `low`

**Honesty rules baked in:**
- If the evidence is thin → it says "Limited evidence"
- If a quote cannot be found in the original text → it is flagged `evidence_unsupported` and confidence is downgraded
- The AI cannot invent reasons — citations must exist in the real profile

**Hidden Gem Detection:**
A candidate is flagged ★ Hidden Gem when:
- Their composite score (Stage 3) is in the top tier
- But their raw embedding rank would have placed them lower
- AND their reason text contains uncertainty language

These are the people a keyword filter would have buried. VIVEKA surfaces them.

---

### Stage 5 — Output (`output.py`)

The final ranked shortlist is written in three formats:

| Format | File | Contents |
|---|---|---|
| Excel | `ranked_output.xlsx` | Formatted spreadsheet — highlighted hidden gems, frozen header, column widths |
| CSV | `ranked_output.csv` | Clean CSV for submission |
| JSON | `ranked_output.json` | Full data including all scores, evidence, audit info |

The Excel file can be downloaded directly from the UI with one click.

---

## Trust & Safety Features

### PII Firewall
Before any AI sees a profile, all personal identity fields are stripped:
name, email, phone, gender, age, location, nationality, photo, address.

Ranking is based on **skills and evidence only**. A human cannot accidentally bias the AI by seeing a candidate's name or photo.

Toggle off: `VIVEKA_PII_FIREWALL=off` (not recommended for real hiring)

### Citation Grounding
Every reason the AI writes must include a verbatim quote from the profile.
A deterministic checker scans the original text to verify each quote exists.
Unverified quotes are flagged — the score is not trusted blindly.

### Keyword Stuffing Penalty
Skills listed in a "Skills" section but never mentioned in actual experience text → 30% penalty on skill score. Honest candidates are not punished for concise writing. Stuffers are caught automatically.

### Full Audit Trail
Every pipeline run is logged to `data/audit.jsonl`.
Each entry contains:
- A SHA-256 run ID (same inputs = same ID, always reproducible)
- All scores and weights used
- The model name
- Every piece of evidence cited

If anyone asks "why did candidate X rank above Y?" — the answer is in the audit log.

---

## The Four Pillars of VIVEKA

VIVEKA is designed as a four-pillar system. **Pillar 4 is fully built and running today.** Pillars 1–3 are the honest roadmap — described here so you can see where this goes, not faked as working features.

### Pillar 1 — Talent Knowledge Graph `[ROADMAP]`
A graph database (Neo4j or similar) that maps skills, roles, and transitions.
It would know: "Spark implies distributed systems", "senior analyst at a fintech likely knows SQL", "ML engineers often come from statistics backgrounds."
This turns isolated résumé facts into a connected map of talent.

### Pillar 2 — Trajectory Modeling `[ROADMAP]`
A model trained on career histories to predict *where a candidate is going*, not just where they've been.
A junior engineer on a steep learning curve ranks higher than a stagnant senior.
Hiring is a bet on the future — this pillar makes that bet smarter.

### Pillar 3 — Candidate Digital Twin `[ROADMAP]`
A profile enrichment layer that infers missing signals from what IS known.
If a profile lists "5 years at a fintech data team" but doesn't mention SQL — SQL is highly probable.
Not invented facts. Probabilistic, honest completion of sparse profiles.

### Pillar 4 — Agentic Recruiter + Ranking `[LIVE NOW]`
The fully working system. An agent that reads a job, understands what it truly needs, ranks candidates, and explains every decision. Runs on free AI models — no paid API required.

---

## The UI — What You See

**Ghost Intelligence Panel**
After ranking, three analytics charts appear automatically:

- **Ghost Score Chart** — "What would each candidate score if they added their missing skills?" Shows potential, not just current fit.
- **Volatility Index** — How stable is each candidate's score across different scoring weights? Low volatility = consistently strong. High volatility = depends on the job type.
- **Score Breakdown** — Side-by-side bar chart of embedding / skill / seniority / activity scores for top candidates. See exactly *why* someone ranked where they did.

**What-If Simulator**
Pick any candidate. Check which skills they're missing. The simulator instantly recalculates their score — showing how much they'd rise in the ranking if they had those skills.

**Leaderboard**
A live ranked table with score, confidence, hidden gem flag, and reason. Top 3 highlighted in gold/silver/bronze. Hidden gems marked with ★.

---

## AI Models Used

VIVEKA supports three AI providers. The system picks automatically based on environment:

| Provider | When Used | Model | Cost |
|---|---|---|---|
| **HuggingFace Inference API** | Default on HF Spaces | Qwen/Qwen2.5-72B-Instruct | Free with HF token |
| **Ollama (local)** | Default on your own machine | Llama 3.2 | Free, offline |
| **Google Gemini** | If `GOOGLE_API_KEY` is set | gemini-1.5-flash | Free tier |

No paid API is required. VIVEKA is designed to run entirely free.

---

## Tech Stack

| What | Tool | Why |
|---|---|---|
| Embeddings | `huggingface_hub InferenceClient` | Free, no GPU needed |
| Fast search | `FAISS` (by Meta) | Finds top 200 from 10,000+ in milliseconds |
| Scoring | `scikit-learn` | Reliable, well-tested math |
| AI reasoning | `LangChain` + HF / Ollama / Gemini | Swap providers in one line |
| Web UI | `Gradio 6` | Clean, fast, runs on HF Spaces natively |
| Excel output | `openpyxl` via `pandas` | Competition-ready formatted spreadsheet |
| Charts | `Plotly` | Interactive, beautiful |
| Terminal output | `Rich` | Color-coded leaderboard in the terminal |
| Privacy | Custom PII firewall | Strips identity before any AI sees the data |
| Audit | Custom JSONL logger | SHA-256 reproducible run IDs |

---

## Project Structure

```
VIVEKA/
  app.py                    ← Entry point — launches Gradio UI
  requirements.txt          ← All dependencies
  .env                      ← Local config (LLM_PROVIDER, model name)
  data/
    job_description.txt     ← Sample job description
    sample_candidates.json  ← 15 demo profiles (no real PII)
    ranked_output.csv       ← Submission CSV
    ranked_output.json      ← Full output with all scores
    ranked_output.xlsx      ← Formatted Excel for submission
    audit.jsonl             ← Audit trail (one line per run)
  src/
    llm.py          ← Model provider (change provider here only)
    agent.py        ← Pipeline orchestrator (runs all 5 stages)
    jd_parser.py    ← Stage 1: JD Intelligence
    recall.py       ← Stage 2: Embedding + FAISS recall
    scoring.py      ← Stage 3: Multi-signal scoring + stuffing detection
    rerank.py       ← Stage 4: LLM rerank + citation grounding
    output.py       ← Stage 5: CSV / JSON / XLSX writer
    gradio_app.py   ← Full Gradio UI (charts, simulator, leaderboard)
    data_loader.py  ← Flexible loader for JSON/CSV with column normalisation
    pii.py          ← PII Firewall
    audit.py        ← Audit trail writer
    config.py       ← All constants and environment variable defaults
    skills.py       ← Skill synonyms (Spark = PySpark = Apache Spark)
```

---

## How to Run

### Option 1 — Live Demo (No Setup)

Just open this link:
**[https://huggingface.co/spaces/mohareddy1423/VIVEKA](https://huggingface.co/spaces/mohareddy1423/VIVEKA)**

Paste a job description, upload or use the sample candidates, click Run.

---

### Option 2 — Run Locally with Ollama (Free, Offline, Private)

```bash
# Step 1: Install Ollama from https://ollama.com

# Step 2: Download the AI model (one time, ~4 GB)
ollama pull llama3.2

# Step 3: Keep Ollama running
ollama serve

# Step 4: Install Python packages
pip install -r requirements.txt

# Step 5: Launch the app
python app.py
# Open http://localhost:7860
```

---

### Option 3 — Run with HuggingFace Inference API (Free, No GPU)

```bash
# Step 1: Get a free HF token from https://huggingface.co/settings/tokens

# Step 2: Set it
# Windows PowerShell:
$env:HF_TOKEN = "hf_..."
$env:LLM_PROVIDER = "huggingface"

# Mac/Linux:
export HF_TOKEN=hf_...
export LLM_PROVIDER=huggingface

# Step 3: Install and run
pip install -r requirements.txt
python app.py
```

---

### Use Your Own Candidate Data

Drop your file into `data/` as `profiles.json` or `profiles.csv`.
VIVEKA auto-detects and normalises common column name variations:
`headline` → `title`, `about` → `summary`, `tech_skills` → `skills`, and many more.

Then run: `python app.py`

---

## Competition Deliverables

| Deliverable | File | Status |
|---|---|---|
| Working code on GitHub | [Mohanreddy-lab/VIVEKA](https://github.com/Mohanreddy-lab/VIVEKA) | ✅ |
| Live demo | [mohareddy1423/VIVEKA on HF Spaces](https://huggingface.co/spaces/mohareddy1423/VIVEKA) | ✅ |
| Ranked output CSV | `data/ranked_output.csv` | ✅ |
| Ranked output XLSX | `data/ranked_output.xlsx` | ✅ |
| Full explanation README | This file | ✅ |
| Audit trail | `data/audit.jsonl` | ✅ |

---

## What We Are NOT Doing (and Why)

**We are not faking Pillars 1–3.**
They are described as the honest roadmap they are. Building fake features that don't work is worse than describing real ones that will.

**We are not inventing performance numbers.**
Every claim in this README is backed by observable outputs from the running pipeline.

**We are not sending candidate data to the cloud without reason.**
The local Ollama mode is the default. Candidate data never leaves your machine unless you choose a cloud provider.

**We are not keyword-matching.**
The entire system is built to defeat keyword matching — in résumés and in hiring tools.

---

## Why VIVEKA Wins on Quality

Most systems show you what candidates *claim*. VIVEKA shows you what candidates can *prove*.

Most systems give you a number. VIVEKA gives you a reason, the evidence behind it, and a confidence level.

Most systems are a black box. VIVEKA has a full audit trail — every decision is reproducible and explainable.

Most systems ignore the overlooked. VIVEKA has a dedicated mechanism to surface hidden gems — strong candidates whose titles or presentation style would have buried them in a keyword filter.

---

## Honesty Statement

> Pillars 1, 2, and 3 are vision. They are described to show where VIVEKA is
> going, not where it is today. Pillar 4 is the working system — fully built,
> tested, and live on HuggingFace Spaces right now.
>
> Every shortlist VIVEKA produces is grounded in real profile text.
> If the evidence is thin, it says so. If a quote cannot be verified, it is flagged.
> If a skill is claimed but not demonstrated, the score is penalized.
>
> **Viveka** — discernment — is not just the name. It is the design principle.
