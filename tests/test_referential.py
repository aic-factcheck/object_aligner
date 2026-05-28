import warnings

import pytest
from jsonschema import ValidationError

from object_aligner import ObjectAligner
from object_aligner.object_aligner import MatchItem


# ---------------------------------------------------------------------------
# Shared schemas
# ---------------------------------------------------------------------------

RELATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "integer", "idScope": "person"},
                    "name": {"type": "string"},
                    "age":  {"type": "integer"},
                },
            },
        },
        "relations": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "integer", "ref": "person"},
                    "target": {"type": "integer", "ref": "person"},
                    "type":   {"type": "string"},
                },
            },
        },
    },
}

GROUPS_SCHEMA = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "integer", "idScope": "person"},
                    "name": {"type": "string"},
                    "age":  {"type": "integer"},
                },
            },
        },
        "research_groups": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "name":    {"type": "string"},
                    "members": {
                        "type": "array",
                        "order": "align",
                        "items": {"type": "integer", "ref": "person"},
                    },
                },
            },
        },
    },
}

PAPERS_SCHEMA = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "integer", "idScope": "person"},
                    "name": {"type": "string"},
                },
            },
        },
        "papers": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "id":      {"type": "integer", "idScope": "paper"},
                    "title":   {"type": "string"},
                    "authors": {
                        "type": "array",
                        "order": "align",
                        "items": {"type": "integer", "ref": "person"},
                    },
                },
            },
        },
        "citations": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "integer", "ref": "paper"},
                    "target": {"type": "integer", "ref": "paper"},
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# 1) Happy-path graphs
# ---------------------------------------------------------------------------

PARENT_CHILD_GOLD = {
    "people": [
        {"id": 1, "name": "Alice", "age": 52},
        {"id": 2, "name": "Bob",   "age": 54},
        {"id": 3, "name": "Cindy", "age": 26},
        {"id": 4, "name": "Dave",  "age": 78},
    ],
    "relations": [
        {"source": 1, "target": 3, "type": "parent_of"},
        {"source": 2, "target": 3, "type": "parent_of"},
        {"source": 3, "target": 1, "type": "child_of"},
        {"source": 3, "target": 2, "type": "child_of"},
        {"source": 4, "target": 1, "type": "parent_of"},
        {"source": 1, "target": 4, "type": "child_of"},
    ],
}

PARENT_CHILD_PRED_RENAMED = {
    "people": [
        {"id": 44,  "name": "Cindy", "age": 26},
        {"id": 53,  "name": "Alice", "age": 52},
        {"id": 99,  "name": "Dave",  "age": 78},
        {"id": 124, "name": "Bob",   "age": 54},
    ],
    "relations": [
        {"source": 53,  "target": 44,  "type": "parent_of"},
        {"source": 124, "target": 44,  "type": "parent_of"},
        {"source": 44,  "target": 53,  "type": "child_of"},
        {"source": 44,  "target": 124, "type": "child_of"},
        {"source": 99,  "target": 53,  "type": "parent_of"},
        {"source": 53,  "target": 99,  "type": "child_of"},
    ],
}


def test_simple_parent_of_child_of_graph_scores_one():
    aligner = ObjectAligner(RELATIONS_SCHEMA)
    assert aligner.metric(PARENT_CHILD_GOLD, PARENT_CHILD_PRED_RENAMED)["score"] == pytest.approx(1.0)


def test_swapped_ids_in_pred_still_one():
    aligner = ObjectAligner(RELATIONS_SCHEMA)
    pred = {
        "people": [
            {"id": 53,  "name": "Alice", "age": 52},
            {"id": 124, "name": "Bob",   "age": 54},
            {"id": 44,  "name": "Cindy", "age": 26},
            {"id": 99,  "name": "Dave",  "age": 78},
        ],
        "relations": PARENT_CHILD_PRED_RENAMED["relations"],
    }
    assert aligner.metric(PARENT_CHILD_GOLD, pred)["score"] == pytest.approx(1.0)


def test_permuted_people_list_still_one():
    aligner = ObjectAligner(RELATIONS_SCHEMA)
    pred = {
        "people": list(reversed(PARENT_CHILD_GOLD["people"])),
        "relations": PARENT_CHILD_GOLD["relations"],
    }
    assert aligner.metric(PARENT_CHILD_GOLD, pred)["score"] == pytest.approx(1.0)


GROUPS_GOLD = {
    "people": [
        {"id": 1, "name": "Alice", "age": 40},
        {"id": 2, "name": "Bob",   "age": 35},
        {"id": 3, "name": "Cindy", "age": 18},
        {"id": 4, "name": "Dave",  "age": 73},
    ],
    "research_groups": [
        {"name": "stem cell", "members": [1, 2, 3]},
        {"name": "quantum",   "members": [2, 3, 4]},
    ],
}

GROUPS_PRED_RENAMED = {
    "people": [
        {"id": 44,  "name": "Cindy", "age": 18},
        {"id": 53,  "name": "Alice", "age": 40},
        {"id": 99,  "name": "Dave",  "age": 73},
        {"id": 124, "name": "Bob",   "age": 35},
    ],
    "research_groups": [
        {"name": "stem cell", "members": [53, 124, 44]},
        {"name": "quantum",   "members": [124, 44, 99]},
    ],
}


def test_multi_graph_members_lists_scores_one():
    aligner = ObjectAligner(GROUPS_SCHEMA)
    assert aligner.metric(GROUPS_GOLD, GROUPS_PRED_RENAMED)["score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2) Dangling refs and integrity errors
# ---------------------------------------------------------------------------

def test_dangling_pred_ref_scores_less_than_one():
    aligner = ObjectAligner(RELATIONS_SCHEMA)
    pred = {
        "people": PARENT_CHILD_PRED_RENAMED["people"],
        "relations": [
            {"source": 53, "target": 9999, "type": "parent_of"},
        ],
    }
    score = aligner.metric(PARENT_CHILD_GOLD, pred)["score"]
    assert 0.0 < score < 1.0


def test_dangling_gold_ref_raises_validation_error():
    aligner = ObjectAligner(RELATIONS_SCHEMA)
    bad_gold = {
        "people": PARENT_CHILD_GOLD["people"],
        "relations": [
            {"source": 1, "target": 9999, "type": "parent_of"},
        ],
    }
    with pytest.raises(ValidationError):
        aligner.metric(bad_gold, PARENT_CHILD_PRED_RENAMED)


def test_duplicate_gold_id_raises_validation_error():
    aligner = ObjectAligner(RELATIONS_SCHEMA)
    bad_gold = {
        "people": [
            {"id": 1, "name": "Alice", "age": 52},
            {"id": 1, "name": "Bob",   "age": 54},
        ],
        "relations": [],
    }
    pred = {"people": [{"id": 5, "name": "X", "age": 1}], "relations": []}
    with pytest.raises(ValidationError):
        aligner.metric(bad_gold, pred)


def test_duplicate_pred_id_first_wins_later_refs_score_zero():
    aligner = ObjectAligner(RELATIONS_SCHEMA)
    gold = {
        "people": [
            {"id": 1, "name": "Alice", "age": 52},
            {"id": 2, "name": "Bob",   "age": 54},
        ],
        "relations": [
            {"source": 1, "target": 2, "type": "knows"},
        ],
    }
    pred = {
        "people": [
            {"id": 10, "name": "Alice", "age": 52},
            {"id": 10, "name": "Bob",   "age": 54},
        ],
        "relations": [
            {"source": 10, "target": 10, "type": "knows"},
        ],
    }
    assert aligner.metric(gold, pred)["score"] < 1.0


# ---------------------------------------------------------------------------
# 3) Bijection size mismatches
# ---------------------------------------------------------------------------

def test_excess_pred_definer_yields_partial_score():
    aligner = ObjectAligner(RELATIONS_SCHEMA)
    pred = {
        "people": PARENT_CHILD_PRED_RENAMED["people"] + [
            {"id": 200, "name": "Eve", "age": 30},
        ],
        "relations": PARENT_CHILD_PRED_RENAMED["relations"],
    }
    score = aligner.metric(PARENT_CHILD_GOLD, pred)["score"]
    assert 0.0 < score < 1.0


def test_missing_pred_definer_yields_partial_score():
    aligner = ObjectAligner(RELATIONS_SCHEMA)
    pred = {
        "people": PARENT_CHILD_PRED_RENAMED["people"][:3],
        "relations": [
            r for r in PARENT_CHILD_PRED_RENAMED["relations"]
            if r["source"] != 99 and r["target"] != 99
        ],
    }
    score = aligner.metric(PARENT_CHILD_GOLD, pred)["score"]
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# 4) Multi-scope cross-scope dependency
# ---------------------------------------------------------------------------

def test_cross_scope_dependency_papers_authors_scores_one():
    aligner = ObjectAligner(PAPERS_SCHEMA)
    gold = {
        "people": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ],
        "papers": [
            {"id": 100, "title": "X", "authors": [1]},
            {"id": 200, "title": "Y", "authors": [1, 2]},
        ],
        "citations": [
            {"source": 100, "target": 200},
        ],
    }
    pred = {
        "people": [
            {"id": 53,  "name": "Alice"},
            {"id": 124, "name": "Bob"},
        ],
        "papers": [
            {"id": 999, "title": "Y", "authors": [124, 53]},
            {"id": 888, "title": "X", "authors": [53]},
        ],
        "citations": [
            {"source": 888, "target": 999},
        ],
    }
    assert aligner.metric(gold, pred)["score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5) Cycle in scope dependency graph
# ---------------------------------------------------------------------------

CYCLE_SCHEMA = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "id":     {"type": "integer", "idScope": "person"},
                    "name":   {"type": "string"},
                    "papers": {
                        "type": "array",
                        "order": "align",
                        "items": {"type": "integer", "ref": "paper"},
                    },
                },
            },
        },
        "papers": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "id":      {"type": "integer", "idScope": "paper"},
                    "title":   {"type": "string"},
                    "authors": {
                        "type": "array",
                        "order": "align",
                        "items": {"type": "integer", "ref": "person"},
                    },
                },
            },
        },
    },
}


def test_cycle_in_scope_dependency_warns_and_completes():
    with pytest.warns(UserWarning, match="Cycle in idScope dependency graph"):
        aligner = ObjectAligner(CYCLE_SCHEMA)

    gold = {
        "people": [{"id": 1, "name": "Alice", "papers": [100]}],
        "papers": [{"id": 100, "title": "X", "authors": [1]}],
    }
    pred = {
        "people": [{"id": 7, "name": "Alice", "papers": [42]}],
        "papers": [{"id": 42, "title": "X", "authors": [7]}],
    }
    score = aligner.metric(gold, pred)["score"]
    assert 0.0 < score <= 1.0


# ---------------------------------------------------------------------------
# 6) Ambiguity warning flag
# ---------------------------------------------------------------------------

TWIN_SCHEMA = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "integer", "idScope": "person"},
                    "name": {"type": "string"},
                    "age":  {"type": "integer"},
                },
            },
        },
        "groups": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "name":    {"type": "string"},
                    "members": {
                        "type": "array",
                        "order": "align",
                        "items": {"type": "integer", "ref": "person"},
                    },
                },
            },
        },
    },
}


# Both Alices are members of the same group: a genuine automorphism that WL
# cannot (and should not) separate — swapping the two Alices is a valid
# isomorphism. This survives WL refinement as residual ambiguity.
TWIN_AUTOMORPHISM_GOLD = {
    "people": [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Alice", "age": 30},
    ],
    "groups": [{"name": "g", "members": [1, 2]}],
}
TWIN_AUTOMORPHISM_PRED = {
    "people": [
        {"id": 10, "name": "Alice", "age": 30},
        {"id": 20, "name": "Alice", "age": 30},
    ],
    "groups": [{"name": "g", "members": [10, 20]}],
}


def test_property_twin_residual_ambiguity_after_wl_emits_warning():
    # Under the WL default, the warning fires only on ambiguity that survives
    # refinement, and its text says so.
    aligner = ObjectAligner(TWIN_SCHEMA, warn_on_ambiguous_mapping=True)
    with pytest.warns(UserWarning, match="Residual ambiguous mapping.*after WL"):
        aligner.metric(TWIN_AUTOMORPHISM_GOLD, TWIN_AUTOMORPHISM_PRED)


def test_property_twin_resolved_by_wl_does_not_warn():
    # Asymmetric reference: only one Alice is in the group, so WL distinguishes
    # them structurally and the (previously fired) warning is now silent.
    aligner = ObjectAligner(TWIN_SCHEMA, warn_on_ambiguous_mapping=True)
    gold = {
        "people": [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Alice", "age": 30},
        ],
        "groups": [{"name": "g", "members": [1]}],
    }
    pred = {
        "people": [
            {"id": 10, "name": "Alice", "age": 30},
            {"id": 20, "name": "Alice", "age": 30},
        ],
        "groups": [{"name": "g", "members": [10]}],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert aligner.metric(gold, pred)["score"] == 1.0


def test_none_strategy_emits_legacy_ambiguity_warning():
    # id_disambiguation="none" reproduces the pre-WL warning text verbatim.
    aligner = ObjectAligner(
        TWIN_SCHEMA, warn_on_ambiguous_mapping=True, id_disambiguation="none"
    )
    gold = {
        "people": [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Alice", "age": 30},
        ],
        "groups": [{"name": "g", "members": [1]}],
    }
    pred = {
        "people": [
            {"id": 10, "name": "Alice", "age": 30},
            {"id": 20, "name": "Alice", "age": 30},
        ],
        "groups": [{"name": "g", "members": [10]}],
    }
    with pytest.warns(UserWarning, match="Ambiguous mapping in idScope"):
        aligner.metric(gold, pred)


def test_warn_on_ambiguous_mapping_off_by_default():
    aligner = ObjectAligner(TWIN_SCHEMA)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aligner.metric(TWIN_AUTOMORPHISM_GOLD, TWIN_AUTOMORPHISM_PRED)


# ---------------------------------------------------------------------------
# 7) Debug tree + description
# ---------------------------------------------------------------------------

def test_debug_tree_includes_ref_marker():
    aligner = ObjectAligner(RELATIONS_SCHEMA)
    result = aligner.metric(PARENT_CHILD_GOLD, PARENT_CHILD_PRED_RENAMED, debug=True)
    debug = result["debug"]

    markers = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("marker"):
                markers.append(node["marker"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(debug)
    assert "ref" in markers
    assert "id" in markers


def test_description_uses_ref_templates_on_mismatch():
    aligner = ObjectAligner(RELATIONS_SCHEMA)
    gold = {
        "people": [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob",   "age": 40},
        ],
        "relations": [
            {"source": 1, "target": 2, "type": "knows"},
        ],
    }
    pred = {
        "people": [
            {"id": 10, "name": "Alice", "age": 30},
            {"id": 20, "name": "Bob",   "age": 40},
        ],
        "relations": [
            {"source": 10, "target": 10, "type": "knows"},
        ],
    }
    result = aligner.metric(gold, pred, generate_description=True)
    assert "inferred id mapping" in result["description"]


# ---------------------------------------------------------------------------
# 8) Backward compatibility
# ---------------------------------------------------------------------------

def test_no_idscope_or_ref_behaves_identically_to_baseline():
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "score": "exact"},
            "y": {"type": "string", "score": "exact"},
        },
    }
    aligner_a = ObjectAligner(schema)
    aligner_b = ObjectAligner(schema)
    s1 = aligner_a.metric({"x": 1, "y": "a"}, {"x": 1, "y": "a"})["score"]
    s2 = aligner_b.metric({"x": 1, "y": "a"}, {"x": 1, "y": "a"})["score"]
    assert s1 == 1.0
    assert s1 == s2
    s3 = aligner_a.metric({"x": 1, "y": "a"}, {"x": 1, "y": "b"})["score"]
    assert 0.0 < s3 < 1.0


# ---------------------------------------------------------------------------
# 9) Construction-time errors
# ---------------------------------------------------------------------------

def test_idscope_on_bool_raises_at_construction():
    bad = {
        "type": "array",
        "order": "align",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "boolean", "idScope": "x"}},
        },
    }
    with pytest.raises(TypeError):
        ObjectAligner(bad)


def test_ref_to_undefined_scope_raises_at_construction():
    bad = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "ref": "ghost"},
        },
    }
    with pytest.raises(ValueError):
        ObjectAligner(bad)


def test_two_idscopes_same_name_raises_at_construction():
    bad = {
        "type": "object",
        "properties": {
            "a": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer", "idScope": "dup"}},
                },
            },
            "b": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer", "idScope": "dup"}},
                },
            },
        },
    }
    with pytest.raises(ValueError):
        ObjectAligner(bad)


def test_type_mismatch_definer_vs_ref_raises_at_construction():
    bad = {
        "type": "object",
        "properties": {
            "people": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer", "idScope": "person"}},
                },
            },
            "names": {
                "type": "array",
                "order": "align",
                "items": {"type": "string", "ref": "person"},
            },
        },
    }
    with pytest.raises(ValueError):
        ObjectAligner(bad)


def test_idscope_outside_any_array_raises_at_construction():
    bad = {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "idScope": "loose"},
        },
    }
    with pytest.raises(ValueError):
        ObjectAligner(bad)


def test_match_item_kind_field_returned_from_align():
    aligner = ObjectAligner(RELATIONS_SCHEMA)
    match = aligner.align(PARENT_CHILD_GOLD, PARENT_CHILD_PRED_RENAMED)
    # Walk the tree and confirm at least one ref-kinded MatchItem with score 1.0
    found_ref = False
    found_id = False

    def walk(node):
        nonlocal found_ref, found_id
        if isinstance(node, MatchItem):
            if node.kind == "ref":
                found_ref = True
            elif node.kind == "id":
                found_id = True
        elif hasattr(node, "children"):
            children = node.children
            if isinstance(children, dict):
                for k, v in children.items():
                    walk(k)
                    walk(v)
            else:
                for c in children:
                    walk(c)

    walk(match)
    assert found_ref
    assert found_id


# ---------------------------------------------------------------------------
# Documented limitation: idScope under JSON Schema composition keywords
# (oneOf/anyOf/allOf/$ref/additionalProperties/patternProperties) is not
# discovered. See docs/referential.md "Limitations".
# ---------------------------------------------------------------------------

def test_idscope_inside_oneof_is_not_discovered():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer", "idScope": "thing"},
                            },
                        }
                    ],
                },
            },
        },
    }
    aligner = ObjectAligner(schema)
    assert aligner._id_scopes == {}


def test_ref_to_idscope_hidden_in_oneof_raises_at_construction():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer", "idScope": "thing"},
                            },
                        }
                    ],
                },
            },
            "refs": {
                "type": "array",
                "items": {"type": "integer", "ref": "thing"},
            },
        },
    }
    with pytest.raises(ValueError, match="undefined idScope 'thing'"):
        ObjectAligner(schema)


# ---------------------------------------------------------------------------
# Concurrency: one aligner instance, many threads.
# ---------------------------------------------------------------------------

def test_align_is_safe_to_call_concurrently_on_one_instance():
    from concurrent.futures import ThreadPoolExecutor

    aligner = ObjectAligner(RELATIONS_SCHEMA)
    baseline = aligner.align(PARENT_CHILD_GOLD, PARENT_CHILD_PRED_RENAMED).score

    def run(_):
        return aligner.align(PARENT_CHILD_GOLD, PARENT_CHILD_PRED_RENAMED).score

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(run, range(32)))

    assert all(r == baseline for r in results)


def test_metric_is_safe_to_call_concurrently_on_one_instance():
    from concurrent.futures import ThreadPoolExecutor

    aligner = ObjectAligner(RELATIONS_SCHEMA)
    baseline = aligner.metric(PARENT_CHILD_GOLD, PARENT_CHILD_PRED_RENAMED)["score"]

    def run(_):
        return aligner.metric(PARENT_CHILD_GOLD, PARENT_CHILD_PRED_RENAMED)["score"]

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(run, range(32)))

    assert all(r == baseline for r in results)


# ---------------------------------------------------------------------------
# 11) Weisfeiler–Leman id disambiguation (id_disambiguation="wl")
# ---------------------------------------------------------------------------

# Attribute-less directed graph: nodes carry only their id, so the bijection
# can only be decided by structure (this is the case the WL feature fixes).
GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer", "idScope": "node"}},
            },
        },
        "edges": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "integer", "ref": "node"},
                    "target": {"type": "integer", "ref": "node"},
                },
            },
        },
    },
}

# Attribute-less membership graph: people referenced from group member lists.
MEMBERSHIP_SCHEMA = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer", "idScope": "person"}},
            },
        },
        "groups": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "members": {
                        "type": "array",
                        "order": "align",
                        "items": {"type": "integer", "ref": "person"},
                    },
                },
            },
        },
    },
}


def _path_graph(ids):
    return {
        "nodes": [{"id": i} for i in ids],
        "edges": [{"source": a, "target": b} for a, b in zip(ids, ids[1:])],
    }


def test_reversed_emission_path_is_fixed_by_wl():
    # The motivating bug: a structurally-perfect path whose nodes are emitted
    # in reverse order. WL recovers the structural bijection; "none" degrades
    # to positional (emission-order) alignment.
    gold = _path_graph([0, 1, 2, 3])
    pred = {
        "nodes": [{"id": 3}, {"id": 2}, {"id": 1}, {"id": 0}],
        "edges": [{"source": 0, "target": 1}, {"source": 1, "target": 2}, {"source": 2, "target": 3}],
    }
    wl = ObjectAligner(GRAPH_SCHEMA)
    none = ObjectAligner(GRAPH_SCHEMA, id_disambiguation="none")
    assert wl.metric(gold, pred)["score"] == 1.0
    assert none.metric(gold, pred)["score"] < 1.0


def test_shifted_ids_in_order_unchanged_between_wl_and_none():
    # Structurally correct pred emitted in canonical order: positional and
    # structural alignment agree, so both strategies score 1.0.
    gold = _path_graph([0, 1, 2, 3])
    pred = _path_graph([100, 101, 102, 103])
    wl = ObjectAligner(GRAPH_SCHEMA)
    none = ObjectAligner(GRAPH_SCHEMA, id_disambiguation="none")
    assert wl.metric(gold, pred)["score"] == 1.0
    assert none.metric(gold, pred)["score"] == 1.0


def test_wl_score_is_invariant_to_node_emission_order():
    gold = _path_graph([0, 1, 2, 3])
    pred_a = _path_graph([10, 11, 12, 13])
    pred_b = {
        "nodes": [{"id": 12}, {"id": 10}, {"id": 13}, {"id": 11}],
        "edges": [{"source": 10, "target": 11}, {"source": 11, "target": 12}, {"source": 12, "target": 13}],
    }
    wl = ObjectAligner(GRAPH_SCHEMA)
    assert wl.metric(gold, pred_a)["score"] == wl.metric(gold, pred_b)["score"] == 1.0


def test_directed_cycle_is_score_invariant_under_rotation():
    gold = {
        "nodes": [{"id": 0}, {"id": 1}, {"id": 2}],
        "edges": [{"source": 0, "target": 1}, {"source": 1, "target": 2}, {"source": 2, "target": 0}],
    }
    pred = {
        "nodes": [{"id": 0}, {"id": 1}, {"id": 2}],
        "edges": [{"source": 1, "target": 2}, {"source": 2, "target": 0}, {"source": 0, "target": 1}],
    }
    wl = ObjectAligner(GRAPH_SCHEMA)
    assert wl.metric(gold, pred)["score"] == 1.0


def test_self_loop_and_parallel_edges_align_structurally():
    # Node 0 has a self-loop and a doubled edge to node 1; node 2 is a sink.
    def build(ids):
        a, b, c = ids
        return {
            "nodes": [{"id": a}, {"id": b}, {"id": c}],
            "edges": [
                {"source": a, "target": a},
                {"source": a, "target": b},
                {"source": a, "target": b},
                {"source": b, "target": c},
            ],
        }

    gold = build([0, 1, 2])
    pred = {  # same graph, ids renamed and nodes/edges emitted out of order
        "nodes": [{"id": 7}, {"id": 9}, {"id": 5}],
        "edges": [
            {"source": 5, "target": 9},
            {"source": 5, "target": 5},
            {"source": 9, "target": 7},
            {"source": 5, "target": 9},
        ],
    }
    wl = ObjectAligner(GRAPH_SCHEMA)
    assert wl.metric(gold, pred)["score"] == 1.0


def test_k_ary_membership_resolves_across_emission_order():
    # Two groups of different sizes -> WL separates the size-3 members from the
    # size-2 members even though every person is attribute-less. The pred is
    # laid out so positional ("none") alignment maps the size-3 members onto a
    # mix of both groups, breaking the memberships.
    gold = {
        "people": [{"id": i} for i in (0, 1, 2, 3, 4)],
        "groups": [{"members": [0, 1, 2]}, {"members": [3, 4]}],
    }
    pred = {
        "people": [{"id": i} for i in (10, 11, 12, 13, 14)],
        "groups": [{"members": [12, 13, 14]}, {"members": [10, 11]}],
    }
    wl = ObjectAligner(MEMBERSHIP_SCHEMA)
    none = ObjectAligner(MEMBERSHIP_SCHEMA, id_disambiguation="none")
    assert wl.metric(gold, pred)["score"] == 1.0
    assert none.metric(gold, pred)["score"] < 1.0


def test_six_cycle_vs_two_triangles_warns_residual_under_wl():
    six = {
        "nodes": [{"id": i} for i in range(6)],
        "edges": [{"source": i, "target": (i + 1) % 6} for i in range(6)],
    }
    two_triangles = {
        "nodes": [{"id": i} for i in range(6)],
        "edges": [
            {"source": 0, "target": 1}, {"source": 1, "target": 2}, {"source": 2, "target": 0},
            {"source": 3, "target": 4}, {"source": 4, "target": 5}, {"source": 5, "target": 3},
        ],
    }
    aligner = ObjectAligner(GRAPH_SCHEMA, warn_on_ambiguous_mapping=True)
    with pytest.warns(UserWarning, match="Residual ambiguous mapping.*after WL"):
        aligner.metric(six, two_triangles)


def test_tie_break_does_not_change_distinguishable_alignments():
    # People carry names/ages, so the cost matrix is already determined; the
    # WL tie-break adds only a sub-gap epsilon and must not move anything. The
    # full debug tree is therefore byte-identical to "none".
    wl = ObjectAligner(RELATIONS_SCHEMA)
    none = ObjectAligner(RELATIONS_SCHEMA, id_disambiguation="none")
    wl_out = wl.metric(PARENT_CHILD_GOLD, PARENT_CHILD_PRED_RENAMED, debug=True)
    none_out = none.metric(PARENT_CHILD_GOLD, PARENT_CHILD_PRED_RENAMED, debug=True)
    assert wl_out["score"] == none_out["score"]
    assert wl_out["debug"] == none_out["debug"]


def test_blend_lambda_endpoints():
    gold = _path_graph([0, 1, 2, 3])
    pred = {
        "nodes": [{"id": 3}, {"id": 2}, {"id": 1}, {"id": 0}],
        "edges": [{"source": 0, "target": 1}, {"source": 1, "target": 2}, {"source": 2, "target": 3}],
    }
    none_score = ObjectAligner(GRAPH_SCHEMA, id_disambiguation="none").metric(gold, pred)["score"]
    blend0 = ObjectAligner(GRAPH_SCHEMA, wl_integration="blend", wl_blend_lambda=0.0)
    blend1 = ObjectAligner(GRAPH_SCHEMA, wl_integration="blend", wl_blend_lambda=1.0)
    assert blend0.metric(gold, pred)["score"] == none_score  # lambda=0 -> property-only
    assert blend1.metric(gold, pred)["score"] == 1.0          # lambda=1 -> pure structural


@pytest.mark.parametrize(
    "kwargs",
    [
        {"id_disambiguation": "bogus"},
        {"wl_integration": "bogus"},
        {"wl_rounds": -1},
        {"wl_rounds": "x"},
        {"wl_rounds": True},
        {"wl_blend_lambda": 1.5},
        {"wl_blend_lambda": -0.1},
        {"wl_blend_lambda": "x"},
        {"wl_blend_lambda": float("inf")},
    ],
)
def test_wl_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        ObjectAligner(GRAPH_SCHEMA, **kwargs)
