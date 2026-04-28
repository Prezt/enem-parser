from __future__ import annotations

import json
import logging
import re
import time

import requests

logger = logging.getLogger(__name__)

_DEFAULT_OPTIONS = {
    "num_ctx": 16384,
    "num_predict": 4096,
    "top_p": 0.9,
    "temperature": 0.1,
}

_OLLAMA_URL = "http://localhost:11434/api/generate"


class OllamaClient:
    def __init__(self, base_url: str = _OLLAMA_URL, options: dict | None = None):
        self.base_url = base_url
        self.options = {**_DEFAULT_OPTIONS, **(options or {})}

    def generate(self, prompt: str, model: str, timeout: int = 300) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": self.options,
        }

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.post(self.base_url, json=payload, timeout=timeout)
                if response.status_code != 200:
                    raise RuntimeError(f"Ollama returned HTTP {response.status_code}: {response.text}")
                return response.json()["response"]
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning("Ollama connection error (attempt %d/3), retrying in %ds: %s", attempt + 1, wait, exc)
                time.sleep(wait)

        raise RuntimeError(f"Ollama unreachable after 3 attempts: {last_exc}")


def extract_json(response: str) -> str:
    """
    Robustly extract a JSON array or object from an LLM response.

    Tries in order:
    1. Direct parse (model returned clean JSON)
    2. Strip ```json ... ``` fences
    3. Extract first [...] array
    4. Extract first {...} object and wrap in array
    """
    text = response.strip()

    # 1. Direct parse — normalise to array
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return text
        if isinstance(parsed, dict):
            return f"[{text}]"
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # 3. Extract first JSON array
    array_match = re.search(r"(\[[\s\S]*\])", text)
    if array_match:
        candidate = array_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # 4. Extract first JSON object and wrap
    obj_match = re.search(r"(\{[\s\S]*\})", text)
    if obj_match:
        candidate = obj_match.group(1).strip()
        try:
            json.loads(candidate)
            return f"[{candidate}]"
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from response (first 200 chars): {text[:200]!r}")
