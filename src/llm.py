# llm.py — Central model provider
# Single place to configure which LLM MANTHAN uses.
# Every other module calls get_llm() — never imports a model directly.
#
# Environment variables:
#   LLM_PROVIDER    "ollama" (default) | "gemini"
#   MANTHAN_MODEL   Ollama model name  (default: "llama3.2")
#   GOOGLE_API_KEY  required when LLM_PROVIDER=gemini

import os
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    """Return a configured LangChain chat model.

    Default: Ollama running llama3.2 locally — free, offline, private.
    Override model name with MANTHAN_MODEL env var.
    Switch to Gemini by setting LLM_PROVIDER=gemini + GOOGLE_API_KEY.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        model = os.getenv("MANTHAN_MODEL", "llama3.2")
        print(f"[llm] Ollama  model={model}")
        return ChatOllama(model=model, temperature=0)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set.")
        model = os.getenv("MANTHAN_MODEL", "gemini-1.5-flash")
        print(f"[llm] Gemini  model={model}")
        return ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)

    raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'. Use 'ollama' or 'gemini'.")


# Quick connectivity check — run with: python src/llm.py
if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke("Reply with exactly three words: Ollama is working.")
    print("LLM says:", response.content)
