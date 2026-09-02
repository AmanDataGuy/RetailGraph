"""
Unit tests for src/agent/nodes.py — the pieces that don't need a live
Groq/Neo4j/Qdrant connection: route_query (pure), _try_template (pure),
_validate_cypher (pure), and the ALLOWED_CATEGORIES/ALLOWED_TAGS vocabulary.

Before this file, src/agent/nodes.py had zero test coverage despite being
the LangGraph agent's router and Cypher-template logic — see RUTHLESS_AUDIT.md
Phase 4. These tests lock in the router-misclassification and vocabulary
findings from that audit so they can't silently regress.
"""

import pytest

from src.agent.nodes import (
    route_query,
    _try_template,
    _validate_cypher,
    ALLOWED_CATEGORIES,
    ALLOWED_TAGS,
)


# ── route_query ────────────────────────────────────────────────────────────
class TestRouteQuery:

    def test_filter_routes_to_cypher(self):
        assert route_query({"intent": "filter"}) == "cypher"

    def test_lookup_routes_to_cypher(self):
        assert route_query({"intent": "lookup"}) == "cypher"

    def test_semantic_routes_to_graphrag(self):
        assert route_query({"intent": "semantic"}) == "graphrag"

    def test_hybrid_routes_to_graphrag(self):
        assert route_query({"intent": "hybrid"}) == "graphrag"

    def test_analytics_routes_to_analytics(self):
        assert route_query({"intent": "analytics"}) == "analytics"

    def test_missing_intent_defaults_to_cypher(self):
        assert route_query({}) == "cypher"

    def test_unknown_intent_defaults_to_cypher(self):
        # An intent value outside the 5-item enum falls through to cypher —
        # documents the current behavior, not necessarily the ideal one.
        assert route_query({"intent": "something_new"}) == "cypher"


# ── _try_template ──────────────────────────────────────────────────────────
class TestTryTemplate:

    def test_no_constraints_returns_none(self):
        # Forces the LLM-fallback path in build_cypher when nothing is set.
        assert _try_template({}) is None

    def test_brand_only_returns_brand_lookup_template(self):
        result = _try_template({"brand": "McCormick"})
        assert result is not None
        cypher, params = result
        assert "MADE_BY" in cypher
        assert params == {"brand": "McCormick"}

    def test_category_and_price_returns_bound_params(self):
        result = _try_template({"category": "Beverages", "max_price": 5.0})
        assert result is not None
        cypher, params = result
        assert params["category"] == "Beverages"
        assert params["max_price"] == 5.0
        # Values must be bound params, never interpolated into the string —
        # this is the injection defense the audit confirmed is real.
        assert "Beverages" not in cypher
        assert "5.0" not in cypher

    def test_dietary_tags_produce_indexed_params(self):
        result = _try_template({"dietary_tags": ["vegan", "gluten-free"]})
        assert result is not None
        cypher, params = result
        assert params["tag_0"] == "vegan"
        assert params["tag_1"] == "gluten-free"

    def test_allergen_exclusion_lowercased_param_names(self):
        result = _try_template({"exclude_allergens": ["peanut"]})
        assert result is not None
        cypher, params = result
        assert "NOT EXISTS" in cypher
        assert params["excl_0"] == "peanut"

    def test_apostrophe_in_brand_does_not_break_query(self):
        # The exact "Reese's" case ROADMAP.md flagged as the pre-fix
        # injection risk — must round-trip as a bound param, not break
        # the Cypher string.
        result = _try_template({"brand": "Reese's"})
        assert result is not None
        cypher, params = result
        assert params["brand"] == "Reese's"
        assert "Reese's" not in cypher


# ── _validate_cypher ──────────────────────────────────────────────────────
class TestValidateCypher:

    def test_valid_match_return_passes(self):
        valid, error = _validate_cypher("MATCH (p:Product) RETURN p")
        assert valid is True
        assert error is None

    def test_missing_match_fails(self):
        valid, error = _validate_cypher("RETURN 1")
        assert valid is False
        assert "MATCH" in error

    def test_missing_return_fails(self):
        valid, error = _validate_cypher("MATCH (p:Product)")
        assert valid is False
        assert "RETURN" in error

    def test_delete_blocked(self):
        valid, error = _validate_cypher("MATCH (p:Product) DETACH DELETE p")
        assert valid is False

    def test_create_blocked(self):
        valid, error = _validate_cypher("CREATE (p:Product) RETURN p")
        assert valid is False

    def test_merge_blocked(self):
        valid, error = _validate_cypher("MERGE (p:Product) RETURN p")
        assert valid is False


# ── Vocabulary consistency — verified against the live Neo4j graph ────────
# (MATCH (c:Category) / MATCH (t:DietaryTag), 2026-08-21). An earlier pass
# matched src/extraction/extractor.py's SYSTEM_PROMPT text instead, which
# turned out not to match what's actually in the graph — see the comment
# above ALLOWED_CATEGORIES in nodes.py for the full story.
class TestAllowedVocabulary:

    def test_keto_friendly_now_recognized_not_keto(self):
        # The graph's real DietaryTag is "keto-friendly" (48 products) — a
        # plain "keto" tag does not exist. "keto snacks" previously returned
        # 0 results every time because of this exact mismatch.
        assert "keto-friendly" in ALLOWED_TAGS
        assert "keto" not in ALLOWED_TAGS

    def test_vegetarian_and_halal_recognized(self):
        # Real, live tags (17 and 3 products respectively) that no prior
        # version of this vocabulary included.
        assert "vegetarian" in ALLOWED_TAGS
        assert "halal" in ALLOWED_TAGS

    def test_dead_tags_removed(self):
        # "paleo", "allergen-free", "cruelty-free" are in extractor.py's
        # SYSTEM_PROMPT vocabulary but have zero matching products in the
        # live graph — the model was told about them but never actually
        # used them on this catalog.
        dead = {"paleo", "allergen-free", "cruelty-free"}
        assert ALLOWED_TAGS.isdisjoint(dead)

    def test_vocab_matches_live_graph_categories(self):
        # Must match the live Category nodes exactly, not extractor.py's
        # prompt text — the prompt text doesn't reflect what actually
        # shipped (e.g. it says "Personal Care & Beauty", the graph has
        # "Personal Care"; it says "Bread & Bakery", the graph has
        # "Bakery & Bread"; several prompt categories have zero products).
        expected = {
            "Snacks & Candy", "Coffee & Tea", "Condiments & Sauces",
            "Beverages", "Spices & Seasonings", "Grains, Beans & Legumes",
            "Unknown", "Supplements & Health", "Personal Care",
            "Meat & Seafood", "Fruits & Vegetables", "Baby & Kids",
            "Pet Supplies", "Frozen Foods", "Dairy & Eggs", "Bakery & Bread",
            "Household & Cleaning",
        }
        assert ALLOWED_CATEGORIES == expected
