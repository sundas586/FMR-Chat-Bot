"""Gemini-based embedding function for ChromaDB, avoiding the heavy local ONNX model."""

from __future__ import annotations

import os
import time

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from google.genai.errors import ClientError


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Compute chunk embeddings via the Gemini API instead of a local ONNX model."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001") -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def __call__(self, input: Documents) -> Embeddings:
        texts = list(input)
        batch_size = 20  # Smaller batches stay well under the free-tier RPM limit.
        embeddings: Embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings.extend(self._embed_batch_with_retry(batch))
            if start + batch_size < len(texts):
                time.sleep(6)  # Pace requests to stay under the free-tier rate limit.
        return embeddings

    def _embed_batch_with_retry(self, batch: list[str], max_attempts: int = 5) -> Embeddings:
        """Call the Gemini embedding API, retrying with backoff on rate-limit errors."""
        delay = 10
        for attempt in range(1, max_attempts + 1):
            try:
                result = self._client.models.embed_content(model=self._model, contents=batch)
                return [item.values for item in result.embeddings]
            except ClientError as exc:
                if exc.code != 429 or attempt == max_attempts:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("Gemini embedding retries exhausted.")

    @staticmethod
    def name() -> str:
        return "gemini"


def get_embedding_function() -> EmbeddingFunction | None:
    """Return a Gemini embedding function when configured, else None for the Chroma default."""
    if os.getenv("LLM_PROVIDER", "ollama").lower() != "gemini":
        return None
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    return GeminiEmbeddingFunction(api_key=api_key, model=model)
