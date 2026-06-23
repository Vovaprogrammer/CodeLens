"""
LLM клиент для CodeLens RAG.
Поддерживает два провайдера с переключением в UI:
  - Ollama (локально, mistral:7b)
  - OpenRouter (облачно, бесплатный tier: mistral-7b-instruct)
"""

import os
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional


SYSTEM_PROMPT = """Ты — ассистент для анализа кода. Пользователь задаёт вопрос о проекте,
а ты получаешь релевантные фрагменты кода, найденные поиском.
Твоя задача: дать краткий, точный ответ на русском или английском (в зависимости от языка вопроса),
опираясь ТОЛЬКО на предоставленные фрагменты.
Не придумывай то, чего нет в коде. Если ответ неизвестен из фрагментов — так и скажи."""


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {}) or {}
        file_path = chunk.get("file_path") or meta.get("file_path", "unknown")
        name = chunk.get("name") or meta.get("name", "unknown")
        content = chunk.get("content", "")
        parts.append(f"[{i}] {file_path} → {name}\n```python\n{content}\n```")
    return "\n\n".join(parts)


def _build_messages(query: str, chunks: List[Dict[str, Any]]) -> List[Dict]:
    context = _build_context(chunks)
    user_msg = f"Вопрос: {query}\n\nНайденные фрагменты кода:\n{context}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


# ──────────────────────────────────────────────
# Ollama
# ──────────────────────────────────────────────

def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def ollama_generate(
    query: str,
    chunks: List[Dict[str, Any]],
    model: str = "mistral:7b",
    base_url: str = "http://localhost:11434",
) -> str:
    messages = _build_messages(query, chunks)
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 1024},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("message", {}).get("content", "").strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"Ollama недоступен: {e}") from e


# ──────────────────────────────────────────────
# OpenRouter
# ──────────────────────────────────────────────

OPENROUTER_MODELS = {
    "mistral-7b (free)":       "mistralai/mistral-7b-instruct:free",
    "llama-3-8b (free)":       "meta-llama/llama-3-8b-instruct:free",
    "gemma-2-9b (free)":       "google/gemma-2-9b-it:free",
    "deepseek-r1 (free)":      "deepseek/deepseek-r1-0528:free",
}


def openrouter_generate(
    query: str,
    chunks: List[Dict[str, Any]],
    api_key: str,
    model_label: str = "mistral-7b (free)",
) -> str:
    model_id = OPENROUTER_MODELS.get(model_label, "mistralai/mistral-7b-instruct:free")
    messages = _build_messages(query, chunks)
    payload = json.dumps({
        "model": model_id,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1024,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/Vovaprogrammer/CodeLens",
            "X-Title": "CodeLens RAG",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"OpenRouter error: {e}") from e


# ──────────────────────────────────────────────
# Единый интерфейс
# ──────────────────────────────────────────────

class LLMClient:
    """
    Единый клиент LLM. Провайдер выбирается в Streamlit UI.
    provider: "ollama" | "openrouter"
    """

    def __init__(
        self,
        provider: str = "ollama",
        ollama_model: str = "mistral:7b",
        ollama_url: str = "http://localhost:11434",
        openrouter_api_key: Optional[str] = None,
        openrouter_model_label: str = "mistral-7b (free)",
    ):
        self.provider = provider
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.openrouter_model_label = openrouter_model_label

    def generate(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        if self.provider == "ollama":
            return ollama_generate(query, chunks, self.ollama_model, self.ollama_url)
        elif self.provider == "openrouter":
            if not self.openrouter_api_key:
                raise RuntimeError("OpenRouter API key не задан. Укажи его в боковой панели.")
            return openrouter_generate(query, chunks, self.openrouter_api_key, self.openrouter_model_label)
        else:
            raise ValueError(f"Неизвестный провайдер LLM: {self.provider}")

    def is_available(self) -> bool:
        if self.provider == "ollama":
            return check_ollama_available(self.ollama_url)
        elif self.provider == "openrouter":
            return bool(self.openrouter_api_key)
        return False
