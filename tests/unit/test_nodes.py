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
    build_cypher,
    execute_query,
    format_answer,
    FORMAT_SYSTEM,
    ANALYTICS_FORMAT_SYSTEM,
    _try_template,
    _validate_cypher,
    _where_scoped_to_optional_match,
    _has_no_filter_condition,
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
        # build_cypher treats this as "nothing to search on" for non-lookup
        # intents rather than falling back to the LLM (see TestBuildCypher).
        assert _try_template({}) is None

    def test_brand_only_returns_brand_lookup_template(self):
        result = _try_template({"brand": "McCormick"})
        assert result is not None
        cypher, params, count_cypher = result
        assert "MADE_BY" in cypher
        assert params == {"brand": "McCormick"}
        assert "count(DISTINCT p)" in count_cypher
        assert "LIMIT" not in count_cypher

    def test_category_and_price_returns_bound_params(self):
        result = _try_template({"category": "Beverages", "max_price": 5.0})
        assert result is not None
        cypher, params, count_cypher = result
        assert params["category"] == "Beverages"
        assert params["max_price"] == 5.0
        # Values must be bound params, never interpolated into the string —
        # this is the injection defense the audit confirmed is real.
        assert "Beverages" not in cypher
        assert "5.0" not in cypher
        assert "count(DISTINCT p)" in count_cypher

    def test_dietary_tags_produce_indexed_params(self):
        result = _try_template({"dietary_tags": ["vegan", "gluten-free"]})
        assert result is not None
        cypher, params, count_cypher = result
        assert params["tag_0"] == "vegan"
        assert params["tag_1"] == "gluten-free"

    def test_allergen_exclusion_lowercased_param_names(self):
        result = _try_template({"exclude_allergens": ["peanut"]})
        assert result is not None
        cypher, params, count_cypher = result
        assert "NOT EXISTS" in cypher
        assert params["excl_0"] == "peanut"

    def test_apostrophe_in_brand_does_not_break_query(self):
        # The exact "Reese's" case ROADMAP.md flagged as the pre-fix
        # injection risk — must round-trip as a bound param, not break
        # the Cypher string.
        result = _try_template({"brand": "Reese's"})
        assert result is not None
        cypher, params, count_cypher = result
        assert params["brand"] == "Reese's"
        assert "Reese's" not in cypher

    def test_brand_combined_with_category_is_actually_filtered(self):
        # Bug found via evaluation/deepeval_results.json + code read: when
        # brand was combined with category/tags, the dynamic query builder
        # never bound $brand into a WHERE clause — it only checked
        # `if ... or brand` to decide whether to return a template at all,
        # so "gluten-free products from McCormick" silently ignored the
        # brand and matched gluten-free products from every brand.
        result = _try_template({"category": "Condiments & Sauces", "brand": "McCormick"})
        assert result is not None
        cypher, params, count_cypher = result
        assert params["brand"] == "McCormick"
        assert "toLower(b.name) = toLower($brand)" in cypher
        assert "toLower(b.name) = toLower($brand)" in count_cypher

    def test_brand_combined_with_tags_is_actually_filtered(self):
        result = _try_template({"dietary_tags": ["vegan"], "brand": "Reese's"})
        assert result is not None
        cypher, params, count_cypher = result
        assert params["brand"] == "Reese's"
        assert "toLower(b.name) = toLower($brand)" in cypher

    def test_price_only_filter_where_precedes_optional_match(self):
        # Bug found live against real Neo4j during stress testing: a WHERE
        # clause placed after an OPTIONAL MATCH scopes to that optional
        # pattern only and never excludes rows — confirmed empirically that
        # "products under $3" (price filter, no brand/tags/category) was
        # returning completely unfiltered results because the WHERE always
        # landed after the unconditional OPTIONAL MATCH category/brand
        # clauses. No part of the generated query may have WHERE appear
        # after any OPTIONAL MATCH.
        result = _try_template({"max_price": 3.0})
        assert result is not None
        cypher, params, count_cypher = result
        assert not _where_scoped_to_optional_match(cypher)
        assert not _where_scoped_to_optional_match(count_cypher)

    def test_allergen_only_filter_where_precedes_optional_match(self):
        result = _try_template({"exclude_allergens": ["peanut"]})
        assert result is not None
        cypher, params, count_cypher = result
        assert not _where_scoped_to_optional_match(cypher)

    def test_brand_plus_price_no_tags_brand_is_mandatory_match(self):
        # The specific combination that was still broken after the first
        # brand fix: brand + price, no tags/category, so nothing else
        # happened to follow the brand match and "absorb" the WHERE scope.
        result = _try_template({"brand": "McCormick", "max_price": 10.0})
        assert result is not None
        cypher, params, count_cypher = result
        assert "MATCH (p)-[:MADE_BY]->(b:Brand)" in cypher
        assert "OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)" not in cypher
        assert not _where_scoped_to_optional_match(cypher)
        assert not _where_scoped_to_optional_match(count_cypher)


# ── _where_scoped_to_optional_match ─────────────────────────────────────────
class TestWhereScopedToOptionalMatch:

    def test_where_after_optional_match_detected(self):
        cypher = (
            "MATCH (p:Product)\n"
            "OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)\n"
            "WHERE p.price <= 3.0\n"
            "RETURN p"
        )
        assert _where_scoped_to_optional_match(cypher) is True

    def test_where_after_mandatory_match_not_flagged(self):
        cypher = (
            "MATCH (p:Product)\n"
            "MATCH (p)-[:HAS_TAG]->(:DietaryTag {name: 'vegan'})\n"
            "WHERE p.price <= 3.0\n"
            "OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)\n"
            "RETURN p"
        )
        assert _where_scoped_to_optional_match(cypher) is False

    def test_no_where_not_flagged(self):
        assert _where_scoped_to_optional_match("MATCH (p:Product) RETURN p") is False


# ── _has_no_filter_condition ─────────────────────────────────────────────────
class TestHasNoFilterCondition:

    def test_bare_match_with_only_optional_enrichment_has_no_filter(self):
        cypher = (
            "MATCH (p:Product) "
            "OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand) "
            "OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category) "
            "RETURN p LIMIT 10"
        )
        assert _has_no_filter_condition(cypher) is True

    def test_where_clause_counts_as_filter(self):
        cypher = "MATCH (p:Product) WHERE p.price <= 5 RETURN p"
        assert _has_no_filter_condition(cypher) is False

    def test_inline_constraint_on_mandatory_match_counts_as_filter(self):
        cypher = "MATCH (p:Product)-[:MADE_BY]->(b:Brand {name: 'Heinz'}) RETURN p"
        assert _has_no_filter_condition(cypher) is False

    def test_inline_constraint_on_optional_match_does_not_count(self):
        # An inline constraint on an OPTIONAL MATCH still doesn't exclude
        # non-matching rows (same null-don't-exclude problem) — only
        # non-optional MATCH constraints count as a real filter.
        cypher = "MATCH (p:Product) OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand {name: 'Heinz'}) RETURN p"
        assert _has_no_filter_condition(cypher) is True


# ── build_cypher ────────────────────────────────────────────────────────────
class TestBuildCypher:

    def test_no_entities_filter_intent_short_circuits_no_llm_call(self, monkeypatch):
        # Bug found in evaluation/deepeval_results.json (q28: "asdkjfh qwerty
        # products xyz123", ground truth 0 matches) — with zero entities
        # extracted, build_cypher used to fall back to LLM-generated Cypher,
        # which returned an arbitrary top-10-by-price result set instead of
        # "no matches" for a query with no real signal. "filter" intent with
        # no entities must now resolve to zero results without ever calling
        # the LLM.
        def _boom(*a, **k):
            raise AssertionError("generate_json should not be called for a no-entity filter query")
        monkeypatch.setattr("src.agent.nodes.generate_json", _boom)

        result = build_cypher({
            "query": "asdkjfh qwerty products xyz123",
            "intent": "filter",
            "entities": {},
        })
        assert result["cypher_query"] is None
        assert result["cypher_valid"] is True
        assert result["raw_results"] == []
        assert result["result_count"] == 0
        assert result["total_count"] == 0

    def test_no_entities_lookup_intent_still_calls_llm(self, monkeypatch):
        # "lookup" intent (a specific product/brand by name) has no
        # dedicated entity field to template against, so it still needs the
        # LLM fallback — this must NOT be short-circuited.
        called = {}
        def _fake_generate_json(system, prompt):
            called["hit"] = True
            return {"cypher": "MATCH (p:Product) RETURN p LIMIT 10"}
        monkeypatch.setattr("src.agent.nodes.generate_json", _fake_generate_json)

        result = build_cypher({
            "query": "tell me about the Solely banana bar",
            "intent": "lookup",
            "entities": {},
        })
        assert called.get("hit") is True
        assert result["cypher_query"] == "MATCH (p:Product) RETURN p LIMIT 10"

    def test_template_match_carries_count_query(self):
        result = build_cypher({
            "query": "vegan snacks under $10",
            "intent": "filter",
            "entities": {"dietary_tags": ["vegan"], "max_price": 10.0},
        })
        assert result["cypher_query"] is not None
        assert result["cypher_count_query"] is not None
        assert "count(DISTINCT p)" in result["cypher_count_query"]


# ── execute_query ──────────────────────────────────────────────────────────
class TestExecuteQuery:

    def test_no_cypher_returns_nonempty_dict(self):
        # Regression: LangGraph requires every node to write at least one
        # state key — returning {} here crashed every zero-entity query with
        # "Must write to at least one of [...]" (500 Internal Server Error),
        # only caught by running the actual containerized stack end to end,
        # not by any unit test in isolation.
        result = execute_query({
            "cypher_query": None,
            "raw_results": [],
            "result_count": 0,
            "total_count": 0,
        })
        assert result != {}
        assert result["raw_results"] == []
        assert result["result_count"] == 0
        assert result["total_count"] == 0


# ── format_answer ──────────────────────────────────────────────────────────
class TestFormatAnswer:

    def _capture_generate(self, monkeypatch):
        captured = {}
        def _fake_generate(system, prompt):
            captured["system"] = system
            captured["prompt"] = prompt
            return "stub answer"
        monkeypatch.setattr("src.agent.nodes.generate", _fake_generate)
        return captured

    def test_truncated_results_get_count_aware_phrasing(self, monkeypatch):
        # Bug found in evaluation/deepeval_results.json (q1: "vegan
        # gluten-free snacks under $10", 21 actual matches, answer said
        # "Found 10 products" — the LIMIT-10 page size was reported to the
        # LLM as "Total results found", so it undercounted every time more
        # than 10 products matched.
        captured = self._capture_generate(monkeypatch)
        state = {
            "query": "vegan gluten-free snacks under $10",
            "raw_results": [{"item_name": f"Item {i}", "price": 1.0} for i in range(10)],
            "result_count": 10,
            "total_count": 21,
        }
        format_answer(state)
        assert "Total matching products: 21 (showing top 10)" in captured["prompt"]
        assert "Total results found: 10" not in captured["prompt"]

    def test_untruncated_results_use_plain_count(self, monkeypatch):
        captured = self._capture_generate(monkeypatch)
        state = {
            "query": "McCormick spices",
            "raw_results": [{"item_name": "Item 1", "price": 1.0}],
            "result_count": 1,
            "total_count": 1,
        }
        format_answer(state)
        assert "Total results found: 1" in captured["prompt"]

    def test_analytics_uses_analytics_system_prompt(self, monkeypatch):
        # Bug found in evaluation/deepeval_results.json (q17-q19): analytics
        # answers with factually correct numbers scored 0.0 on correctness,
        # all sharing FORMAT_SYSTEM's product-search framing ("Found X
        # products matching your search") applied to aggregate rows.
        captured = self._capture_generate(monkeypatch)
        state = {
            "query": "which category has the most products",
            "raw_results": [{"category": "Snacks & Candy", "product_count": 580}],
            "result_count": 1,
        }
        format_answer(state)
        assert captured["system"] == ANALYTICS_FORMAT_SYSTEM
        assert captured["system"] != FORMAT_SYSTEM

    def test_product_results_use_product_system_prompt(self, monkeypatch):
        captured = self._capture_generate(monkeypatch)
        state = {
            "query": "vegan snacks",
            "raw_results": [{"item_name": "Item 1", "price": 1.0}],
            "result_count": 1,
        }
        format_answer(state)
        assert captured["system"] == FORMAT_SYSTEM

    def test_empty_results_never_calls_llm(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("generate should not be called for empty results")
        monkeypatch.setattr("src.agent.nodes.generate", _boom)

        result = format_answer({"query": "xyz", "raw_results": [], "result_count": 0})
        assert "No products found" in result["answer"]


# ── _validate_cypher ──────────────────────────────────────────────────────
class TestValidateCypher:

    def test_valid_match_return_passes(self):
        valid, error = _validate_cypher("MATCH (p:Product) WHERE p.price <= 5 RETURN p")
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

    def test_where_after_optional_match_rejected(self):
        cypher = (
            "MATCH (p:Product)\n"
            "OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)\n"
            "WHERE toLower(p.item_name) CONTAINS toLower('x')\n"
            "RETURN p"
        )
        valid, error = _validate_cypher(cypher)
        assert valid is False
        assert "OPTIONAL MATCH" in error

    def test_unconstrained_query_rejected(self):
        # Bug found live: for a "lookup" query with no identifiable search
        # term, the LLM sometimes generates a query with NO filter at all
        # (just the base match plus enrichment OPTIONAL MATCHes) — this
        # silently returned an arbitrary top-10 slice of the whole catalog
        # instead of recognizing there was nothing to search for.
        cypher = (
            "MATCH (p:Product) "
            "OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand) "
            "OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category) "
            "RETURN p.item_name, p.price LIMIT 10"
        )
        valid, error = _validate_cypher(cypher)
        assert valid is False
        assert "filters nothing" in error

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
