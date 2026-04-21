# object-aligner

Python library for aligning structured objects (`dict`, `list`, primitives) and scoring similarity in `[0, 1]` with human-readable reasoning.

## Install

```bash
uv add object-aligner
```

Or:

```bash
pip install object-aligner
```

## Quick example

```python
from object_aligner import ObjectAligner

schema = {"type": "string", "score": "jaro"}
aligner = ObjectAligner("demo", schema)

result = aligner.metric("hello", "hallo")
print(result["score"])
print(result["reasoning"])
```

## Development

Install dependencies:

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```

The repository includes a comprehensive pytest suite under `tests/` covering primitives, lists, dicts, nesting, API behavior, regressions from the docs, and edge cases.

## Documentation

See `docs/index.md` for the full guide.
