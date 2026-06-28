# llm.py — Central model provider
# Single place to configure which LLM VIVEKA uses.
# Every other module calls get_llm() — never imports a model directly.
#
# Environment variables:
#   LLM_PROVIDER    "ollama" (default) | "gemini"
#   VIVEKA_MODEL    Ollama model name  (default: "llama3.2")
#   GOOGLE_API_KEY  required when LLM_PROVIDER=gemini

import os
import logging
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("viveka.llm")

DEFAULT_MODEL  = "llama3.2"
DEFAULT_HOST   = "http://localhost:11434"


def get_llm(json_mode: bool = False):
    """Return a configured LangChain chat model.

    Default: Ollama running llama3.2 locally — free, offline, private.
    json_mode=True forces JSON output (Ollama only) — prevents small models
    from outputting markdown or prose instead of JSON.

    Env vars:
      VIVEKA_MODEL    — model name (default: llama3.2)
      OLLAMA_HOST     — Ollama server URL (default: http://localhost:11434)
      LLM_PROVIDER    — "ollama" (default) | "gemini"
      GOOGLE_API_KEY  — required when LLM_PROVIDER=gemini
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        model = os.getenv("VIVEKA_MODEL", DEFAULT_MODEL)
        host  = os.getenv("OLLAMA_HOST",   DEFAULT_HOST)
        kwargs: dict = {"model": model, "temperature": 0, "base_url": host}
        if json_mode:
            kwargs["format"] = "json"
        log.info("Ollama model=%s json_mode=%s host=%s", model, json_mode, host)
        return ChatOllama(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not set. Add it to your .env file or environment."
            )
        model = os.getenv("VIVEKA_MODEL", "gemini-1.5-flash")
        log.info("Gemini model=%s", model)
        return ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)

    raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'. Use 'ollama' or 'gemini'.")


def check_ollama() -> tuple[bool, str]:
    """Check Ollama is running and the configured model is available.
    Returns (ok, message).
    """
    import requests
    model = os.getenv("VIVEKA_MODEL", DEFAULT_MODEL)
    host  = os.getenv("OLLAMA_HOST",  DEFAULT_HOST)
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        if r.status_code != 200:
            return False, "Ollama server returned non-200 status. Run: ollama serve"
        available = [m["name"] for m in r.json().get("models", [])]
        model_base = model.split(":")[0]
        found = any(model in m or m.startswith(model_base) for m in available)
        if not found:
            avail_str = ", ".join(available) or "none"
            return False, (
                f"Model '{model}' not pulled.\n"
                f"Run: ollama pull {model}\n"
                f"Available: {avail_str}"
            )
        return True, f"Ollama OK — {model} ready"
    except requests.exceptions.ConnectionError:
        return False, "Ollama is not running.\nStart it: ollama serve"
    except Exception as exc:
        return False, f"Ollama check failed: {exc}"


# Quick connectivity check — run with: python src/llm.py
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok, msg = check_ollama()
    print(f"Health: {'✅' if ok else '❌'} {msg}")
    if ok:
        llm = get_llm()
        response = llm.invoke("Reply with exactly three words: Ollama is working.")
        print("LLM says:", response.content)
