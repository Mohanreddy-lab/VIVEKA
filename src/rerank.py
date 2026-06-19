# rerank.py — Stage 4: Honest LLM Rerank
# For the top ~50 candidates, asks GPT-4o to score fit and write a short
# reason grounded in real profile text. Never invents evidence — says so
# if the proof is weak.
