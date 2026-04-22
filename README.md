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

Built-in string metrics include `exact`, `jaro`, `jaro_winkler`, `levenshtein`, `damerau_levenshtein`, `osa`, `indel`, and `lcsseq`.

## Custom named metrics

Schemas stay declarative: put the metric name in `score`, and pass the implementation through `custom_metrics` when you construct the aligner.

```python
from object_aligner import ObjectAligner


def semantic_toy(gold: str, pred: str) -> float:
    return 0.95 if gold.lower()[0] == pred.lower()[0] else 0.2


schema = {"type": "string", "score": "semantic_toy", "threshold": 0.5}
aligner = ObjectAligner(
    schema,
    custom_metrics={"string": {"semantic_toy": semantic_toy}},
)

print(aligner.metric("cat", "car"))
print(aligner.metric("cat", "dog"))
```

Numeric schemas support the same mechanism:

```python

def closish(gold: float, pred: float) -> float:
    return 0.8 if abs(gold - pred) <= 2 else 0.1


aligner = ObjectAligner(
    {"type": "number", "score": "closish"},
    custom_metrics={"number": {"closish": closish}},
)
```

For integer schemas, custom metrics first look in the `integer` registry and then fall back to the `number` registry.

## API note

`ObjectAligner` takes the schema plus optional keyword arguments such as `custom_metrics` and `generate_reasoning`:

```python
aligner = ObjectAligner(schema)
aligner = ObjectAligner(schema, custom_metrics={"string": {...}})
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
