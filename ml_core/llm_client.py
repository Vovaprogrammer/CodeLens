"""
CodeLens RAG — Клиент для OpenRouter API.
Оставлена только автоматическая бесплатная модель.
"""

import os
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional


SYSTEM_PROMPT = """Ты — ИИ-ассистент для анализа кода в системе CodeLens. 
Помогай разработчику понять логику проекта на основе предоставленных фрагментов кода.

Твои правила:
1. Отвечай строго на языке вопроса (если спросили на русском — отвечай на русском, если на английском — на английском).
2. Опирайся только на предоставленный в запросе код. Если информации в чанках нет, вежливо ответь, что в данных фрагментах логика не найдена.
3. Форматируй ответ с помощью Markdown, чтобы его было удобно читать:

### 🎯 Суть ответа
(Краткий и понятный ответ на вопрос)

### ⚙️ Разбор логики
(Пояснение, как это устроена в коде, с упоминанием названий файлов или функций из контекста)
"""


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


def openrouter_generate(
    query: str,
    chunks: List[Dict[str, Any]],
    api_key: str,
) -> str:
    messages = _build_messages(query, chunks)
    payload = json.dumps({
        "model": "openrouter/free",  # Всегда автомодель
        "messages": messages,
        "temperature": 0.1,
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
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if e.code == 429:
            raise RuntimeError(
                "Превышены лимиты запросов (HTTP 429) на бесплатном тарифе OpenRouter. "
                "Пожалуйста, подождите 5-10 секунд и повторите попытку."
            )
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"Ошибка сети OpenRouter: {e}") from e


class LLMClient:
    """Единый изолированный клиент OpenRouter AI Автомодели."""

    def __init__(self, openrouter_api_key: Optional[str] = None):
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")

    def generate(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        if not self.openrouter_api_key:
            raise RuntimeError("API-ключ OpenRouter не задан. Пожалуйста, укажите его в боковой панели.")
        return openrouter_generate(query, chunks, self.openrouter_api_key)

    def is_available(self) -> bool:
        return bool(self.openrouter_api_key)