# 5. Nesting & Complex Structures

The real power of Object Aligner emerges with deeply nested data. Since every alignment method recursively calls `_align_helper`, you can combine objects, arrays, and primitives at arbitrary depth.

This chapter walks through realistic examples of increasing complexity.

---

## Example 1: List of simple objects

A common pattern in LLM evaluation: the model extracts a list of entities, each described by a small dict.

```python
from object_aligner import ObjectAligner

gold = [
    {"name": "Alice", "score": 95},
    {"name": "Bob",   "score": 82}
]
pred = [
    {"name": "Alice", "score": 93},
    {"name": "Bobby", "score": 82}
]

schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name":  {"type": "string", "score": "jaro", "valueWeight": 1.0},
            "score": {"type": "integer", "score": "invdiff", "valueWeight": 1.0}
        },
        "keyScore": "exact",
        "keyImportance": 0.0,
        "valueImportance": 1.0
    },
    "order": "align"  # order doesn't matter — find best pairing
}

aligner = ObjectAligner(schema, generate_reasoning=True)
result = aligner.metric(gold, pred)
print(f"Score: {result['score']:.2f}")
print(result["reasoning"])
```

The Hungarian algorithm pairs `{"name": "Alice"}` with `{"name": "Alice"}` and `{"name": "Bob"}` with `{"name": "Bobby"}`, then recursively scores each pair.

---

## Example 2: Product catalog entry

A product record with a name, price, tags (unordered), and specs (key-value dict).

```python
gold = {
    "product": "Wireless Headphones",
    "price": 79.99,
    "tags": ["bluetooth", "noise-cancelling", "over-ear"],
    "specs": {
        "battery_life": 30,
        "weight_grams": 250,
        "driver_size_mm": 40
    }
}

pred = {
    "product": "Wireless Headphone",
    "price": 74.99,
    "tags": ["blutooth", "noise-canceling", "over-ear", "foldable"],
    "specs": {
        "battery_life": 28,
        "weight_grams": 255,
        "driver_size_mm": 40
    }
}

schema = {
    "type": "object",
    "properties": {
        "product": {
            "type": "string",
            "score": "jaro",
            "threshold": 0.5,
            "valueWeight": 2.0
        },
        "price": {
            "type": "number",
            "score": "invdiff",
            "threshold": 0.0,
            "valueWeight": 3.0
        },
        "tags": {
            "type": "array",
            "items": {
                "type": "string",
                "score": "jaro",
                "threshold": 0.5
            },
            "order": "align",
            "ignoreExcess": True,
            "valueWeight": 1.0
        },
        "specs": {
            "type": "object",
            "properties": {
                "battery_life":  {"type": "integer", "score": "invdiff", "valueWeight": 2.0},
                "weight_grams":  {"type": "integer", "score": "invdiff", "valueWeight": 1.0},
                "driver_size_mm": {"type": "integer", "score": "exact",  "valueWeight": 1.0}
            },
            "keyScore": "exact",
            "keyImportance": 0.0,
            "valueImportance": 1.0,
            "valueWeight": 1.0
        }
    },
    "keyScore": "exact",
    "keyImportance": 0.0,
    "valueImportance": 1.0
}

aligner = ObjectAligner(schema, generate_reasoning=True)
result = aligner.metric(gold, pred)
print(f"Score: {result['score']:.2f}")
print(result["reasoning"])
```

Key design decisions in this schema:

| Decision | Rationale |
|----------|-----------|
| `product` has `valueWeight: 2.0` | Product name matters more than tags |
| `price` has `valueWeight: 3.0` | Price accuracy is critical for e-commerce |
| `tags` uses `order: "align"` + `ignoreExcess: True` | Tag order is irrelevant; extra tags are OK |
| `specs.battery_life` has `valueWeight: 2.0` | Battery life is the headline spec |
| `specs.driver_size_mm` uses `score: "exact"` | Driver size must match exactly |

---

## Example 3: Nested lists — restaurant orders

Each order is a list of items, each item is `[dish_name, quantity, notes...]`:

```python
gold = [
    ["Margherita", 2, "extra cheese"],
    ["Pepperoni", 1, "thin crust", "no onions"]
]
pred = [
    ["Margharita", 2, "extra cheesse"],
    ["Pepperoni", 1, "thin crust"]
]

schema = {
    "type": "array",
    "items": {
        "type": "array",
        "prefixItems": [
            {"type": "string"},   # dish name
            {"type": "integer"}   # quantity
        ],
        "prefixWeights": [2, 3],       # quantity matters more than dish name spelling
        "items": {"type": "string"},   # notes
        "prefixImportance": 3.0,       # prefix (name+qty) is 3x more important
        "restImportance": 1.0          # notes are less critical
    },
    "order": "fixed"
}

aligner = ObjectAligner(schema, generate_reasoning=True)
result = aligner.metric(gold, pred)
print(f"Score: {result['score']:.2f}")
print(result["reasoning"])
```

This example demonstrates how `prefixItems` + `items` combine in nested arrays. The dish name and quantity (prefix) dominate the score, while missing notes ("no onions") have a smaller impact.

---

## Example 4: Deeply nested structure — exam results

A school report: each student has subjects, each subject has a grade and comments.

```python
gold = [
    {
        "student": "Emma Johnson",
        "subjects": [
            {"subject": "Mathematics", "grade": 92, "comments": ["excellent", "hardworking"]},
            {"subject": "History", "grade": 85, "comments": ["good analysis"]}
        ]
    },
    {
        "student": "Liam Smith",
        "subjects": [
            {"subject": "Mathematics", "grade": 78, "comments": ["improving"]},
            {"subject": "Physics", "grade": 88, "comments": ["strong practical work"]}
        ]
    }
]

pred = [
    {
        "student": "Emma Jonson",
        "subjects": [
            {"subject": "Math", "grade": 90, "comments": ["excellent", "hardwarking"]},
            {"subject": "History", "grade": 84, "comments": ["good analysys"]}
        ]
    },
    {
        "student": "Liam Smith",
        "subjects": [
            {"subject": "Mathematics", "grade": 78, "comments": ["improving"]},
            {"subject": "Physic", "grade": 90, "comments": ["strong practical"]}
        ]
    }
]

schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "student": {"type": "string", "score": "jaro", "valueWeight": 2.0},
            "subjects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject":  {"type": "string", "score": "jaro", "valueWeight": 2.0},
                        "grade":    {"type": "integer", "score": "invdiff", "valueWeight": 3.0},
                        "comments": {
                            "type": "array",
                            "items": {"type": "string", "score": "jaro"},
                            "order": "align",
                            "ignoreExcess": True,
                            "ignoreMissing": True,
                            "valueWeight": 1.0
                        }
                    },
                    "keyScore": "exact",
                    "keyImportance": 0.0,
                    "valueImportance": 1.0
                },
                "order": "align",
                "valueWeight": 3.0
            }
        },
        "keyScore": "exact",
        "keyImportance": 0.0,
        "valueImportance": 1.0
    },
    "order": "align"
}

aligner = ObjectAligner(schema, generate_reasoning=True)
result = aligner.metric(gold, pred)
print(f"Score: {result['score']:.2f}")
print(result["reasoning"])
```

Design highlights:

- **Top-level** `order: "align"` pairs students by best overall match
- **Subjects** also use `order: "align"` — "Mathematics" in gold pairs with "Math" in pred via Jaro
- **Comments** use `ignoreExcess` + `ignoreMissing` so slight differences don't kill the score
- **Grades** use `invdiff` with high `valueWeight` — getting close to the right grade matters most

---

## Tips for designing nested schemas

1. **Start from the top** and work inward. Define the outermost type first, then add `properties` or `items`.

2. **Use `keyImportance: 0`** when you care about values but not key matching quality (common when using `"keyScore": "exact"`).

3. **Use `order: "align"`** for any collection where position is not meaningful.

4. **Tune `valueWeight`** to reflect which fields matter most for your use case. If price accuracy is critical, give it a higher weight.

5. **Set thresholds** to prevent garbage pairings. A `keyThreshold` of 0.5 prevents "age" from matching "title".

6. **Use `ignoreExcess`/`ignoreMissing`** for open-ended lists where over- or under-generation shouldn't be heavily penalized.
