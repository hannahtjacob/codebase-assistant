"""Replaceable language-model providers for answer generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:1.5b"


class LLMProvider(Protocol):
    """Dependency-inversion boundary for text generation."""

    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class OllamaProvider:
    """Generate answers through a locally running Ollama server."""

    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str = "http://127.0.0.1:11434"
    timeout: float = 120.0

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300},
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(
                "Cannot reach Ollama. Start it with `ollama serve` and ensure "
                f"the {self.model!r} model is installed."
            ) from error

        answer = body.get("response")
        if not isinstance(answer, str):
            raise RuntimeError("Ollama response did not contain generated text")
        return answer.strip()
