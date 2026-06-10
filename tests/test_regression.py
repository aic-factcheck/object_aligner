import pytest

from object_aligner import ObjectAligner


def test_primitives_documented_examples():
    assert ObjectAligner({"type": "boolean"}).metric(True, False)["score"] == 0.0
    assert ObjectAligner({"type": "integer", "score": "invdiff", "threshold": 0.5}).metric(50, 51)["score"] == 0.5
    assert ObjectAligner({"type": "integer", "score": "invdiff", "threshold": 0.5}).metric(50, 52)["score"] == 0.0
    assert ObjectAligner({"type": "string", "score": "jaro"}).metric("hello", "hallo")["score"] > 0.8


def test_lists_documented_examples():
    quiz = ObjectAligner({"type": "array", "items": {"type": "integer", "score": "exact"}, "order": "fixed"},
    )
    skills = ObjectAligner({"type": "array", "items": {"type": "string", "score": "jaro", "threshold": 0.5}, "order": "align"},
    )

    assert quiz.metric([42, 7, 13], [99, 7, 13])["score"] == pytest.approx(0.5)
    assert skills.metric(["Python", "JavaScript", "SQL"], ["Pythn", "SQL", "JavaScrypt"])["score"] == pytest.approx(0.9592592592592593)
    assert skills.metric(["weight", "name", "age"], ["name", "ages", "title"])["score"] == pytest.approx(0.47916666666666663)


def test_dicts_documented_examples():
    exact = ObjectAligner({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "score": "invdiff"},
            },
            "keyScore": "exact",
        },
    )
    fuzzy = ObjectAligner({
            "type": "object",
            "properties": {
                "weight": {"type": "integer", "valueWeight": 1.0},
                "name": {"type": "string", "score": "jaro", "valueWeight": 1.0},
                "age": {"type": "integer", "valueWeight": 1.0},
            },
            "keyScore": "jaro",
            "keyThreshold": 0.5,
            "keyImportance": 1.0,
            "valueImportance": 1.0,
        },
    )

    # Default keyImportance=0 (exact schema): score is the mean of value pairs.
    assert exact.metric({"name": "Alice", "age": 30}, {"name": "Alicia", "age": 29})["score"] == pytest.approx(0.6611111111111112)
    # fuzzy schema pins keyImportance=1 explicitly.
    assert fuzzy.metric({"weight": 90, "name": "John", "age": 24}, {"name": "Johny", "ages": 23, "title": "Mr."})["score"] == pytest.approx(0.41874999999999996)


def test_nesting_documented_examples():
    students = ObjectAligner({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "score": "jaro", "valueWeight": 1.0},
                    "score": {"type": "integer", "score": "invdiff", "valueWeight": 1.0},
                },
                "keyScore": "exact",
                "keyImportance": 0.0,
                "valueImportance": 1.0,
            },
            "order": "align",
        },
    )
    product = ObjectAligner({
            "type": "object",
            "properties": {
                "product": {"type": "string", "score": "jaro", "threshold": 0.5, "valueWeight": 2.0},
                "price": {"type": "number", "score": "invdiff", "threshold": 0.0, "valueWeight": 3.0},
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "score": "jaro", "threshold": 0.5},
                    "order": "align",
                    "ignoreExcess": True,
                    "valueWeight": 1.0,
                },
                "specs": {
                    "type": "object",
                    "properties": {
                        "battery_life": {"type": "integer", "score": "invdiff", "valueWeight": 2.0},
                        "weight_grams": {"type": "integer", "score": "invdiff", "valueWeight": 1.0},
                        "driver_size_mm": {"type": "integer", "score": "exact", "valueWeight": 1.0},
                    },
                    "keyScore": "exact",
                    "keyImportance": 0.0,
                    "valueImportance": 1.0,
                    "valueWeight": 1.0,
                },
            },
            "keyScore": "exact",
            "keyImportance": 0.0,
            "valueImportance": 1.0,
        },
    )
    orders = ObjectAligner({
            "type": "array",
            "items": {
                "type": "array",
                "prefixItems": [{"type": "string"}, {"type": "integer"}],
                "prefixWeights": [2, 3],
                "items": {"type": "string"},
                "prefixImportance": 3.0,
                "restImportance": 1.0,
            },
            "order": "fixed",
        },
    )
    exam = ObjectAligner({
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
                                "subject": {"type": "string", "score": "jaro", "valueWeight": 2.0},
                                "grade": {"type": "integer", "score": "invdiff", "valueWeight": 3.0},
                                "comments": {
                                    "type": "array",
                                    "items": {"type": "string", "score": "jaro"},
                                    "order": "align",
                                    "ignoreExcess": True,
                                    "valueWeight": 1.0,
                                },
                            },
                            "keyScore": "exact",
                            "keyImportance": 0.0,
                            "valueImportance": 1.0,
                        },
                        "order": "align",
                        "valueWeight": 3.0,
                    },
                },
                "keyScore": "exact",
                "keyImportance": 0.0,
                "valueImportance": 1.0,
            },
            "order": "align",
        },
    )

    assert students.metric(
        [{"name": "Alice", "score": 95}, {"name": "Bob", "score": 82}],
        [{"name": "Alice", "score": 93}, {"name": "Bobby", "score": 82}],
    )["score"] == pytest.approx(0.8)
    assert product.metric(
        {
            "product": "Wireless Headphones",
            "price": 79.99,
            "tags": ["bluetooth", "noise-cancelling", "over-ear"],
            "specs": {"battery_life": 30, "weight_grams": 250, "driver_size_mm": 40},
        },
        {
            "product": "Wireless Headphone",
            "price": 74.99,
            "tags": ["blutooth", "noise-canceling", "over-ear", "foldable"],
            "specs": {"battery_life": 28, "weight_grams": 255, "driver_size_mm": 40},
        },
    )["score"] == pytest.approx(0.557707927225471)
    assert orders.metric(
        [["Margherita", 2, "extra cheese"], ["Pepperoni", 1, "thin crust", "no onions"]],
        [["Margharita", 2, "extra cheesse"], ["Pepperoni", 1, "thin crust"]],
    )["score"] == pytest.approx(0.9131837606837607)
    assert exam.metric(
        [
            {
                "student": "Emma Johnson",
                "subjects": [
                    {"subject": "Mathematics", "grade": 92, "comments": ["excellent", "hardworking"]},
                    {"subject": "History", "grade": 85, "comments": ["good analysis"]},
                ],
            },
            {
                "student": "Liam Smith",
                "subjects": [
                    {"subject": "Mathematics", "grade": 78, "comments": ["improving"]},
                    {"subject": "Physics", "grade": 88, "comments": ["strong practical work"]},
                ],
            },
        ],
        [
            {
                "student": "Emma Jonson",
                "subjects": [
                    {"subject": "Math", "grade": 90, "comments": ["excellent", "hardwarking"]},
                    {"subject": "History", "grade": 84, "comments": ["good analysys"]},
                ],
            },
            {
                "student": "Liam Smith",
                "subjects": [
                    {"subject": "Mathematics", "grade": 78, "comments": ["improving"]},
                    {"subject": "Physic", "grade": 90, "comments": ["strong practical"]},
                ],
            },
        ],
    )["score"] == pytest.approx(0.8399336774336774)


def test_adversarial_and_boundary_regressions():
    reorder = ObjectAligner({"type": "array", "items": {"type": "string", "score": "exact"}, "order": "align"},
    )
    fixed = ObjectAligner({"type": "array", "items": {"type": "string", "score": "exact"}, "order": "fixed"},
    )
    numeric = ObjectAligner({"type": "number", "score": "invdiff"})
    threshold = ObjectAligner({"type": "string", "score": "jaro", "threshold": 0.7777777777777777})

    values = [f"v{i}" for i in range(100)]
    assert reorder.metric(values, list(reversed(values)))["score"] == 1.0
    assert fixed.metric(values, values)["score"] == 1.0
    assert threshold.metric("cat", "car")["score"] == pytest.approx(0.7777777777777777)
    assert numeric.metric(0, 1_000_000)["score"] < 1e-5
    assert numeric.metric(1.0, 1.001)["score"] > 0.999

    bool_aligner = ObjectAligner({"type": "boolean"})
    assert bool_aligner.metric(True, False)["score"] == 0.0
