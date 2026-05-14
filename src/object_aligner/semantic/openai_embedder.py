"""OpenAI-compatible embedding client.

Talks to any server that implements the OpenAI ``POST /v1/embeddings``
shape — including the cloud OpenAI service, ``llama-cpp`` started with
``--embedding``, ``vllm`` serving an embedding model, and several other
inference runtimes. Uses the official ``openai`` Python SDK; install it
with::

    pip install object-aligner[semantic-openai]

The SDK is imported **lazily** inside ``__init__``, so
``from object_aligner.semantic import OpenAIEmbedder`` always succeeds —
the missing-extra error is raised at construction time with a clear
remediation message.

Cache-key namespacing
---------------------
:attr:`OpenAIEmbedder.model_id` baked from ``base_url`` host + port,
the ``model`` string, and the ``dimensions`` parameter (if supplied).
Two deployments serving the same model name therefore receive different
cache namespaces and cannot pollute each other.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import numpy as np

from object_aligner.semantic.embedder import BaseEmbedder


_MISSING_EXTRA_MSG = (
    "OpenAIEmbedder requires the 'semantic-openai' extra. "
    "Install with: pip install object-aligner[semantic-openai]"
)


class OpenAIEmbedder(BaseEmbedder):
    """Embed strings via any OpenAI-compatible ``/v1/embeddings`` endpoint.

    Args:
        model: The model name to request. For cloud OpenAI: e.g.
            ``"text-embedding-3-small"``. For ``llama-cpp`` on
            localhost: typically the served model name, e.g.
            ``"Qwen3-Embedding-4B"``.
        base_url: Endpoint root URL. Default ``"https://api.openai.com/v1"``.
            For local servers pass e.g. ``"http://localhost:8333/v1"``
            (or ``"http://localhost:8333"`` — the SDK accepts both).
        api_key: Bearer token. Resolution order:
            (1) explicit argument, (2) ``OPENAI_API_KEY`` environment
            variable, (3) the literal string ``"not-needed"`` (most
            local servers ignore the header). The SDK refuses ``None``,
            so we always pass *something*.
        dimensions: Optional ``dimensions`` request parameter (cloud
            OpenAI only). When set, included in :attr:`model_id` for
            cache-key safety.
        max_batch_size: Maximum number of texts per HTTP call. Defaults
            to ``256``. OpenAI cloud allows up to 2048; ``llama-cpp`` is
            usually configured for fewer. Longer input lists are
            chunked internally and the responses are concatenated.
        timeout: Per-request timeout in seconds. Passed to the SDK.
        max_retries: Number of automatic retries on transient errors.
            Delegated to the SDK's built-in retry policy.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        dimensions: int | None = None,
        max_batch_size: int = 256,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        if not isinstance(model, str) or not model:
            raise ValueError(f"model must be a non-empty string, got {model!r}")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError(f"base_url must be a non-empty string, got {base_url!r}")
        if not isinstance(max_batch_size, int) or max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be a positive int, got {max_batch_size!r}"
            )

        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(_MISSING_EXTRA_MSG) from e

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"

        self._model = model
        self._base_url = base_url
        self._dimensions = dimensions
        self._max_batch_size = max_batch_size
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = OpenAI(
            base_url=base_url,
            api_key=resolved_key,
            timeout=timeout,
            max_retries=max_retries,
        )

        host = urlparse(base_url).netloc or base_url
        dim_token = f"d={dimensions}" if dimensions is not None else "d=default"
        self._model_id = f"{host}::{model}::{dim_token}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int | None:
        return self._dimensions

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        for t in texts:
            if not isinstance(t, str):
                raise TypeError(
                    f"OpenAIEmbedder only embeds str, got {type(t).__name__}"
                )

        out: list[np.ndarray] = []
        for i in range(0, len(texts), self._max_batch_size):
            chunk = texts[i : i + self._max_batch_size]
            kwargs: dict[str, Any] = {"model": self._model, "input": chunk}
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions
            response = self._client.embeddings.create(**kwargs)
            # The SDK returns the data list in input order.
            for item in response.data:
                out.append(np.asarray(item.embedding, dtype=np.float32))
        if len(out) != len(texts):
            raise RuntimeError(
                f"OpenAIEmbedder: backend returned {len(out)} vectors "
                f"for {len(texts)} inputs"
            )
        return out
