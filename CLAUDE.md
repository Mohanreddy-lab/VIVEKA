# VIVEKA — Discerning true talent from noise

## The vision
VIVEKA discerns the best candidate for a job — not by matching keywords,
but by separating genuine signal from résumé noise, ranking by evidence,
and explaining every choice. Like viveka (विवेक) — the wisdom to tell
true from false, the real from the surface. Built for the "India Runs"
Data & AI Challenge.

## Runs fully offline, free
VIVEKA uses a LOCAL language model (Ollama + Llama 3.2), so it needs
NO paid API and NO internet. Candidate data never leaves the machine.
This is a feature: privacy-first hiring.

## What we must deliver (3 things, scored)
1. A clean GitHub repo with working code.
2. A README that explains the full vision and our decisions.
3. A ranked output file in the EXACT format the organizers ask for.

## The full VIVEKA system (4 pillars)
Pillar 1 - Talent Knowledge Graph  ........ ROADMAP (describe, don't build)
Pillar 2 - Trajectory modeling  ........... ROADMAP (describe, don't build)
Pillar 3 - Candidate Digital Twin  ........ ROADMAP (describe, don't build)
Pillar 4 - Agentic Recruiter + Ranking  ... BUILD THIS (the working core)

We BUILD Pillar 4 fully. We DESCRIBE pillars 1-3 in the README as
the future roadmap. We never fake them as working.

## What we build now: Pillar 4 (the engine)
An agent that reads a job, figures out what it really needs, ranks
candidates, and explains the result. Inside it runs 5 stages:

1. JD Intelligence: read the job. Pull out needed skills,
   hidden/implied skills, and seniority. Save as structured data.
2. Fast recall: turn the job + all profiles into embeddings.
   Use FAISS to quickly grab the top ~200 closest candidates.
3. Multi-signal scoring: score them by mixing
   - meaning match (embedding similarity)
   - skill overlap (weighted by importance)
   - any activity/behavior signals in the data.
4. Honest rerank: for the top ~50, ask the LOCAL model to score
   fit and write a short reason using REAL text from the profile.
   If proof is weak, it must say so. Never invent reasons.
5. Output: write the ranked list in the organizers' format, with
   score, reason, and a "hidden gem" flag for strong but
   overlooked candidates.

The "agent" wraps these 5 stages: it decomposes the job, runs the
pipeline, and presents the shortlist with clear explanations.

## Tech stack (all free)
- Python
- sentence-transformers (embeddings, local, free)
- FAISS (fast search, local, free)
- scikit-learn (scoring)
- LangChain + Ollama running Llama 3.2 (local model, free)
- Streamlit (simple demo UI)
- Optional cloud backup: Gemini free tier (only if needed)

## Project structure
viveka/
  app.py           # Streamlit entry-point (HF Spaces + local)
  data/            # dataset goes here
  src/
    llm.py         # returns the language model (Ollama by default)
    agent.py       # Pillar 4: decomposes job, runs pipeline, explains
    data_loader.py # flexible .json/.csv loader with column normalisation
    jd_parser.py   # Stage 1
    recall.py      # Stage 2
    scoring.py     # Stage 3
    rerank.py      # Stage 4
    output.py      # Stage 5
    pii.py         # PII firewall
    audit.py       # Audit trail
    demo.py        # Streamlit app
  README.md
  requirements.txt
  CLAUDE.md

## Coding rules
- Keep functions small and clearly named.
- Add short comments in plain English.
- Build ONE stage at a time. Test before moving on.
- After each working stage, make a git commit.
- All model choices come from src/llm.py, so we can swap
  Ollama <-> Gemini in ONE place.
- Read any settings (model name) from environment variables.
- NOTE: env var names use MANTHAN_ prefix (e.g. MANTHAN_MODEL,
  MANTHAN_RERANK_N). Do NOT rename them — they are stable API surface.

## Honesty rules (these win trust)
- Do NOT build pillars 1-3. Only describe them in the README.
- Do NOT put made-up money/value numbers anywhere.
- Prove the mission with a REAL hidden-gem example instead.

## Dataset note
We do not have the dataset yet. Keep data loading flexible.
When the real dataset arrives, match the loader and output
format to the organizers' spec exactly.
