"""
app.py — VIVEKA Gradio entry-point for Hugging Face Spaces and local runs.

LLM modes (set via HF Spaces Secrets or .env):
  huggingface (default on HF Spaces) — LLM_PROVIDER=huggingface
      Uses HF Inference API. HF_TOKEN is auto-set on Spaces — no setup needed.
      Default model: Qwen/Qwen2.5-72B-Instruct  (change via VIVEKA_MODEL)

  ollama (local default)             — LLM_PROVIDER=ollama
      Requires Ollama running locally. Free, private, offline.

  gemini                             — LLM_PROVIDER=gemini  GOOGLE_API_KEY=<key>
      Google Gemini free tier.

Run locally:
  python app.py
  # or: gradio app.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from gradio_app import demo, _THEME, CSS  # noqa: E402

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # needed inside Docker
        server_port=int(os.environ.get("PORT", 7860)),
        theme=_THEME,
        css=CSS,
    )
