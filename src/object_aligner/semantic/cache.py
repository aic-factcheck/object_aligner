"""Embedding caches.

The cache is the middle layer of the semantic stack: it wraps an
:class:`~object_aligner.semantic.Embedder` and adds memoisation plus
miss-batching. Two concrete classes ship:

* :class:`InMemoryEmbeddingCache` — ``dict``-backed, no I/O. Fast and
  perfectly adequate for single-process workloads up to ~10⁵ entries.
* :class:`SQLiteEmbeddingCache` — backed by a SQLite database via the
  standard library. Persistent across processes, safe under concurrent
  writers (WAL journal mode), no new dependencies.

Both classes inherit the same ``get_many`` semantics:

1. Probe storage for each input text in order.
2. De-duplicate misses (the same string appearing twice in the input
   list counts once).
3. Call ``self.embedder.embed_many(unique_misses)`` exactly once per
   ``get_many`` call.
4. Store the new vectors.
5. Reassemble the result list in the original input order.

Cache keys are ``(model_id, text)`` tuples. ``model_id`` comes from the
embedder; if a user swaps embedders, old vectors are filtered out by
the namespacing so stale results never leak.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from object_aligner.semantic.embedder import Embedder


class EmbeddingCache:
    """Abstract base for the two ``get_many`` semantics.

    Subclasses implement :meth:`_lookup` and :meth:`_store`. The
    ``get_many`` / ``get_one`` / ``__contains__`` / ``__len__`` /
    ``clear`` surfaces are shared.

    Attributes:
        embedder: The wrapped :class:`Embedder` whose ``embed_many`` is
            invoked on cache miss.
    """

    embedder: Embedder

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    # ----- Subclass hooks ---------------------------------------------

    def _lookup(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Return the subset of ``texts`` already cached under
        ``self.embedder.model_id``. Subclasses override."""
        raise NotImplementedError

    def _store(self, items: dict[str, np.ndarray]) -> None:
        """Persist new ``{text: vector}`` items under
        ``self.embedder.model_id``. Subclasses override."""
        raise NotImplementedError

    def _all_texts(self) -> list[str]:
        """Return the list of cached texts for the current model_id.
        Subclasses override; used by ``__len__`` and ``__contains__``.
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Remove all cached entries for the current model_id."""
        raise NotImplementedError

    # ----- Shared API -------------------------------------------------

    def get_one(self, text: str) -> np.ndarray:
        return self.get_many([text])[0]

    def get_many(self, texts: list[str]) -> list[np.ndarray]:
        """Resolve embeddings for ``texts`` (hit cache, miss-batch).

        Args:
            texts: Input strings, possibly with duplicates. Order is
                preserved in the returned list.

        Returns:
            One ``np.ndarray`` per input text, in input order. Cached
            entries are returned by reference; newly embedded ones are
            stored before return.
        """
        if not texts:
            return []
        hits = self._lookup(texts)
        unique_misses: list[str] = []
        seen: set[str] = set()
        for t in texts:
            if t in hits or t in seen:
                continue
            seen.add(t)
            unique_misses.append(t)

        if unique_misses:
            fresh = self.embedder.embed_many(unique_misses)
            if len(fresh) != len(unique_misses):
                raise RuntimeError(
                    f"embedder.embed_many returned {len(fresh)} vectors "
                    f"for {len(unique_misses)} inputs"
                )
            new_items = dict(zip(unique_misses, fresh))
            self._store(new_items)
            hits.update(new_items)

        return [hits[t] for t in texts]

    def __contains__(self, text: str) -> bool:
        return text in set(self._all_texts())

    def __len__(self) -> int:
        return len(self._all_texts())


# ---------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------

class InMemoryEmbeddingCache(EmbeddingCache):
    """``dict``-backed cache; no I/O.

    Backing store is ``{(model_id, text): np.ndarray}``. Use this for
    single-process workloads, tests, or as a fast inner layer wrapped
    by an outer persistent cache (not provided here).
    """

    def __init__(self, embedder: Embedder):
        super().__init__(embedder)
        self._store_dict: dict[tuple[str, str], np.ndarray] = {}

    def _lookup(self, texts: list[str]) -> dict[str, np.ndarray]:
        mid = self.embedder.model_id
        out: dict[str, np.ndarray] = {}
        for t in texts:
            v = self._store_dict.get((mid, t))
            if v is not None:
                out[t] = v
        return out

    def _store(self, items: dict[str, np.ndarray]) -> None:
        mid = self.embedder.model_id
        for t, v in items.items():
            self._store_dict[(mid, t)] = v

    def _all_texts(self) -> list[str]:
        mid = self.embedder.model_id
        return [t for (m, t) in self._store_dict if m == mid]

    def clear(self) -> None:
        mid = self.embedder.model_id
        for key in [k for k in self._store_dict if k[0] == mid]:
            del self._store_dict[key]


# ---------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS embeddings (
    model_id TEXT NOT NULL,
    text     TEXT NOT NULL,
    dtype    TEXT NOT NULL,
    shape    TEXT NOT NULL,
    payload  BLOB NOT NULL,
    PRIMARY KEY (model_id, text)
);
"""


class SQLiteEmbeddingCache(EmbeddingCache):
    """SQLite-backed persistent cache.

    Storage details:

    * One row per ``(model_id, text)``.
    * ``payload`` is the embedding's raw bytes; ``dtype`` records the
      NumPy dtype string and ``shape`` is a JSON-encoded list of axis
      lengths, so both 1-D (sentence-level) and 2-D (token-level)
      embeddings round-trip cleanly.
    * The connection is opened with ``check_same_thread=False`` so a
      single instance is usable from multiple threads; multiple
      processes may share the file safely.
    * Journal mode defaults to ``WAL`` and ``synchronous`` to
      ``NORMAL``, the standard high-throughput SQLite combo for caches.

    Args:
        embedder: The wrapped embedder.
        path: Filesystem path to the SQLite database. Pass
            ``":memory:"`` for an ephemeral in-memory database (used by
            tests). Parent directories are created if they do not exist
            (unless ``path == ":memory:"``).
        journal_mode: Passed to ``PRAGMA journal_mode``. Default
            ``"WAL"``.
        synchronous: Passed to ``PRAGMA synchronous``. Default
            ``"NORMAL"``.
    """

    def __init__(
        self,
        embedder: Embedder,
        path: str | Path,
        *,
        journal_mode: str = "WAL",
        synchronous: str = "NORMAL",
    ):
        super().__init__(embedder)
        self._path = str(path) if path == ":memory:" else str(Path(path).expanduser())
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        self._conn.execute(f"PRAGMA journal_mode={journal_mode};")
        self._conn.execute(f"PRAGMA synchronous={synchronous};")
        self._conn.execute(_SCHEMA_SQL)

    # ----- Storage hooks ---------------------------------------------

    def _decode(self, dtype: str, shape: str, payload: bytes) -> np.ndarray:
        return np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(json.loads(shape))

    def _lookup(self, texts: list[str]) -> dict[str, np.ndarray]:
        if not texts:
            return {}
        mid = self.embedder.model_id
        # Avoid blowing past SQLite's parameter limit (default ~999):
        # chunk lookups when the input is huge.
        out: dict[str, np.ndarray] = {}
        chunk = 500
        for i in range(0, len(texts), chunk):
            piece = texts[i : i + chunk]
            placeholders = ",".join("?" for _ in piece)
            rows = self._conn.execute(
                f"SELECT text, dtype, shape, payload FROM embeddings "
                f"WHERE model_id=? AND text IN ({placeholders})",
                [mid, *piece],
            ).fetchall()
            for text, dtype, shape, payload in rows:
                out[text] = self._decode(dtype, shape, payload)
        return out

    def _store(self, items: dict[str, np.ndarray]) -> None:
        if not items:
            return
        mid = self.embedder.model_id
        rows = []
        for t, v in items.items():
            arr = np.ascontiguousarray(v)
            rows.append((
                mid, t,
                arr.dtype.str,
                json.dumps(list(arr.shape)),
                arr.tobytes(),
            ))
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (model_id, text, dtype, shape, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )

    def _all_texts(self) -> list[str]:
        mid = self.embedder.model_id
        return [r[0] for r in self._conn.execute(
            "SELECT text FROM embeddings WHERE model_id=?", (mid,)
        ).fetchall()]

    def clear(self) -> None:
        mid = self.embedder.model_id
        self._conn.execute("DELETE FROM embeddings WHERE model_id=?", (mid,))

    # ----- Connection lifecycle --------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection. Idempotent."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):  # pragma: no cover - best-effort
        try:
            if getattr(self, "_conn", None) is not None:
                self._conn.close()
        except Exception:
            pass

    # ----- Convenience -----------------------------------------------

    @property
    def path(self) -> str:
        """The database file path, or ``":memory:"`` for an in-memory DB."""
        return self._path

    def total_rows(self) -> int:
        """Total rows across all model_ids in the database (debug aid)."""
        return int(self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
