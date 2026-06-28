"""
app.py — MANTHAN Streamlit entry-point for Hugging Face Spaces and local runs.

Delegates to src/demo.py after applying a provider guard so the app shows
a friendly setup message instead of crashing when GOOGLE_API_KEY is missing.

LLM modes:
  Local (default):   LLM_PROVIDER=ollama  MANTHAN_MODEL=llama3.2
  Cloud demo:        LLM_PROVIDER=gemini  GOOGLE_API_KEY=<your-free-key>

Run:
  streamlit run app.py
"""

import os
import sys
from pathlib import Path

# Make src/ importable for all downstream imports
_src = Path(__file__).parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import streamlit as st

# ── Provider guard — friendly message instead of a crash ─────────────────────
provider = os.getenv("LLM_PROVIDER", "ollama").lower()

if provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
    st.set_page_config(page_title="MANTHAN — Setup", page_icon="🔍")
    st.title("🔍 MANTHAN — Setup Required")
    st.error(
        "**GOOGLE_API_KEY is not set.**\n\n"
        "This instance is configured to use the Gemini API (cloud mode) "
        "but no API key was found."
    )
    st.markdown("""
**To get a free Gemini key:**
1. Go to https://aistudio.google.com
2. Sign in with a Google account → click **Get API key**
3. Copy the key and set it:

**Hugging Face Spaces** — go to *Settings → Repository secrets* and add `GOOGLE_API_KEY`.

**Local** — add to your `.env` file:
```
LLM_PROVIDER=gemini
GOOGLE_API_KEY=AIza...your-key...
```

---
**Or switch to fully local mode (no key needed):**
```
LLM_PROVIDER=ollama
MANTHAN_MODEL=llama3.2
```
Install Ollama at https://ollama.com, then: `ollama pull llama3.2`
    """)
    st.stop()

# ── Delegate to the full demo ────────────────────────────────────────────────
# exec() runs demo.py in this script's global namespace, which is exactly
# what Streamlit does on each rerun — keeps all st.* calls in the right context.
_demo_path = _src / "demo.py"
exec(compile(_demo_path.read_text(encoding="utf-8"), str(_demo_path), "exec"), globals())
