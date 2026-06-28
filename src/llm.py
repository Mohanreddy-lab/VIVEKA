# llm.py — Central model provider
# Single place to configure which LLM VIVEKA uses.
# Every other module calls get_llm() — never imports a model directly.
#
# Environment variables:
#   LLM_PROVIDER    "ollama" (default) | "gemini" | "huggingface"
#   VIVEKA_MODEL    model name — defaults per provider below
#   GOOGLE_API_KEY  required when LLM_PROVIDER=gemini
#   HF_TOKEN        auto-set on HF Spaces; optional elsewhere for higher rate limits

import os
import logging
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("viveka.llm")

DEFAULT_MODEL     = "llama3.2"
DEFAULT_HOST      = "http://localhost:11434"
DEFAULT_HF_MODEL  = "Qwen/Qwen2.5-72B-Instruct"


def get_llm(json_mode: bool = False):
    """Return a configured LangChain chat model.

    Providers (set LLM_PROVIDER env var):
      huggingface  — HF Inference API (free, HF_TOKEN auto-set on HF Spaces) [default on Spaces]
      ollama       — local Llama 3.2 via Ollama (free, offline) [default locally]
      gemini       — Google Gemini API (free tier, needs GOOGLE_API_KEY)

    json_mode=True forces JSON output (Ollama only).
    """
    # Auto-detect HF Spaces via SPACE_ID so no manual variable setup is needed.
    _on_spaces   = bool(os.getenv("SPACE_ID"))
    _default     = "huggingface" if _on_spaces else "ollama"
    provider     = os.getenv("LLM_PROVIDER", _default).lower()

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
            if _on_spaces:
                log.warning("LLM_PROVIDER=gemini but GOOGLE_API_KEY not set — falling back to huggingface")
                provider = "huggingface"
            else:
                raise EnvironmentError(
                    "GOOGLE_API_KEY not set. Add it to your .env file or environment."
                )
        else:
            model = os.getenv("VIVEKA_MODEL", "gemini-1.5-flash")
            log.info("Gemini model=%s", model)
            return ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)

    if provider == "huggingface":
        # Uses huggingface_hub.InferenceClient — no transformers/langchain-huggingface needed.
        # HF_TOKEN is auto-injected by HF Spaces; works without a token on public models.
        return _HFInferenceChat(
            model_id=os.getenv("VIVEKA_MODEL", DEFAULT_HF_MODEL),
            hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN", ""),
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: '{provider}'. Use 'ollama', 'gemini', or 'huggingface'."
    )


# ---------------------------------------------------------------------------
# Minimal LangChain-compatible chat model backed by HF Inference API
# Avoids langchain-huggingface (and its transformers dep) entirely.
# ---------------------------------------------------------------------------

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from pydantic import Field  # noqa: E402


class _HFInferenceChat(BaseChatModel):
    """LangChain ChatModel backed by huggingface_hub.InferenceClient (chat completions)."""

    model_id: str = Field(default=DEFAULT_HF_MODEL)
    hf_token: str = Field(default="")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=self.hf_token or None)

        hf_msgs = []
        for m in messages:
            if isinstance(m, SystemMessage):
                hf_msgs.append({"role": "system",    "content": m.content})
            elif isinstance(m, HumanMessage):
                hf_msgs.append({"role": "user",      "content": m.content})
            elif isinstance(m, AIMessage):
                hf_msgs.append({"role": "assistant", "content": m.content})

        resp = client.chat_completion(
            model=self.model_id,
            messages=hf_msgs,
            max_tokens=512,
            temperature=0.1,
        )
        content = resp.choices[0].message.content or ""
        log.info("HFInferenceChat model=%s tokens=%s", self.model_id,
                 getattr(resp, "usage", {}) and resp.usage.total_tokens)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "hf_inference_chat"


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
