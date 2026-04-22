# object-aligner

Python library for aligning structured objects (`dict`, `list`, primitives) and scoring similarity in `[0, 1]`, with optional human-readable reasoning.

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
aligner = ObjectAligner(schema)

result = aligner.metric("hello", "hallo")
print(result)  # {"score": ...}

result_with_reasoning = aligner.metric("hello", "hallo", generate_reasoning=True)
print(result_with_reasoning["score"])
print(result_with_reasoning["reasoning"])
```

## API note

`ObjectAligner` now takes only the schema:

```python
aligner = ObjectAligner(schema)
```

The older `ObjectAligner("name", schema)` form and `get_name()` are no longer supported.

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
