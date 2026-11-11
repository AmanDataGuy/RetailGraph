"""
# test_nodes.py - unit tests for agent/nodes.py that need no live Groq/Neo4j/Qdrant

- Covers routing, Cypher templates and validation, node edge cases and the tag/category vocabulary.
"""

import sys
import types

import pytest
from retailgraph.agent.nodes import (
    ALLOWED_CATEGORIES,
    ALLOWED_TAGS,
    ANALYTICS_FORMAT_SYSTEM,
    FORMAT_SYSTEM,
    _has_no_filter_condition,
    _try_template,
    _validate_cypher,
    _where_scoped_to_optional_match,
    build_cypher,
    execute_query,
    extract_intent,
    format_answer,
    hybrid_search_node,
    route_query,
)
from retailgraph.graph.queries import allergen_terms


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
        # Unknown intents fall through to cypher (current behaviour, not necessarily ideal).
        assert route_query({"intent": "something_new"}) == "cypher"


# ── _try_template ──────────────────────────────────────────────────────────
class TestTryTemplate:

    def test_no_constraints_returns_none(self):
        # build_cypher then returns zero results for non-lookup intents (see TestBuildCypher).
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
        # Values must be bound params, never interpolated (injection defence).
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
        assert params["excl_0"] == ["peanut"]   # substring terms, so "peanuts" also matches

    def test_apostrophe_in_brand_does_not_break_query(self):
        # An apostrophe must round-trip as a bound param, not break the Cypher string.
        result = _try_template({"brand": "Reese's"})
        assert result is not None
        cypher, params, count_cypher = result
        assert params["brand"] == "Reese's"
        assert "Reese's" not in cypher

    def test_brand_combined_with_category_is_actually_filtered(self):
        # brand combined with category/tags must still be bound into a WHERE clause.
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
        # A WHERE after an OPTIONAL MATCH stops filtering rows, so no WHERE may follow one.
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
        # brand + price with no tags/category: the WHERE must still precede every OPTIONAL MATCH.
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
        # An inline constraint on an OPTIONAL MATCH does not exclude rows, so it is not a filter.
        cypher = "MATCH (p:Product) OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand {name: 'Heinz'}) RETURN p"
        assert _has_no_filter_condition(cypher) is True


# ── build_cypher ────────────────────────────────────────────────────────────
class TestBuildCypher:

    def test_no_entities_filter_intent_short_circuits_no_llm_call(self, monkeypatch):
        # "filter" intent with no entities must give zero results without calling the LLM.
        def _boom(*a, **k):
            raise AssertionError("generate_json should not be called for a no-entity filter query")
        monkeypatch.setattr("retailgraph.agent.nodes.generate_json", _boom)

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
        # "lookup" has no entity field to template against, so it must still reach the LLM.
        called = {}
        def _fake_generate_json(system, prompt):
            called["hit"] = True
            return {"cypher": "MATCH (p:Product) RETURN p LIMIT 10"}
        monkeypatch.setattr("retailgraph.agent.nodes.generate_json", _fake_generate_json)

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
        # LangGraph requires every node to write at least one state key; {} would crash.
        # Same state build_cypher returns when there is nothing to search.
        result = execute_query({
            "cypher_query": None,
            "cypher_valid": True,
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
        monkeypatch.setattr("retailgraph.agent.nodes.generate", _fake_generate)
        return captured

    def test_truncated_results_get_count_aware_phrasing(self, monkeypatch):
        # The full match count, not the LIMIT-10 page size, must reach the LLM.
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
        # Aggregate rows need the analytics prompt, not the product-search framing.
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
        monkeypatch.setattr("retailgraph.agent.nodes.generate", _boom)

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
        # A query with no filter at all would return an arbitrary slice of the catalog.
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


# ── Vocabulary: must match the live Neo4j Category/DietaryTag nodes ────────
class TestAllowedVocabulary:

    def test_keto_friendly_now_recognized_not_keto(self):
        # The graph's tag is "keto-friendly"; a plain "keto" tag does not exist.
        assert "keto-friendly" in ALLOWED_TAGS
        assert "keto" not in ALLOWED_TAGS

    def test_vegetarian_and_halal_recognized(self):
        # Real tags with products in the live graph.
        assert "vegetarian" in ALLOWED_TAGS
        assert "halal" in ALLOWED_TAGS

    def test_dead_tags_removed(self):
        # Prompt-only tags with zero products in the live graph.
        dead = {"paleo", "allergen-free", "cruelty-free"}
        assert ALLOWED_TAGS.isdisjoint(dead)

    def test_vocab_matches_live_graph_categories(self):
        # Must match the live Category nodes exactly.
        expected = {
            "Snacks & Candy", "Coffee & Tea", "Condiments & Sauces",
            "Beverages", "Spices & Seasonings", "Grains, Beans & Legumes",
            "Unknown", "Supplements & Health", "Personal Care",
            "Meat & Seafood", "Fruits & Vegetables", "Baby & Kids",
            "Pet Supplies", "Frozen Foods", "Dairy & Eggs", "Bakery & Bread",
            "Household & Cleaning",
        }
        assert ALLOWED_CATEGORIES == expected


# ── Allergen terms and misc node fixes ──────────────────────────────────────
class TestAllergenTerms:

    def test_plural_maps_to_singular_substring(self):
        assert allergen_terms("Peanuts") == ["peanut"]

    def test_umbrella_words_expand(self):
        assert "milk" in allergen_terms("dairy") and "butter" in allergen_terms("dairy")
        assert {"almond", "cashew", "nut"} <= set(allergen_terms("nuts"))
        assert "wheat" in allergen_terms("gluten")

    def test_empty_word_excludes_nothing(self):
        # "" would match every allergen via CONTAINS and wipe all results
        assert allergen_terms("  ") == []


class TestSweepFixes:

    def test_non_gmo_tag_survives_intent_normalisation(self, monkeypatch):
        monkeypatch.setattr("retailgraph.agent.nodes.generate_json",
                            lambda **kw: {"intent": "filter", "dietary_tags": ["non-gmo", "Vegan"]})
        out = extract_intent({"query": "non-gmo vegan snacks"})
        assert out["entities"]["dietary_tags"] == ["non-GMO", "vegan"]

    @pytest.mark.parametrize("bad", [
        "MATCH (p:Product) WHERE p.price > 1 SET p.price = 0 RETURN p",
        "MATCH (p:Product) WHERE p.price > 1 REMOVE p.price RETURN p",
        "LOAD CSV FROM 'x' AS r MATCH (p:Product) WHERE p.price > 1 RETURN p",
        "MATCH (p:Product) WHERE p.price > 1 CALL db.labels() YIELD label RETURN p",
    ])
    def test_write_and_procedure_clauses_blocked(self, bad):
        valid, _ = _validate_cypher(bad)
        assert not valid

    def test_last_invalid_retry_reports_error(self):
        out = execute_query({"cypher_query": "x", "cypher_valid": False,
                             "cypher_retries": 1, "cypher_error": "bad"})
        assert out["cypher_retries"] == 2 and out["error"] and out["raw_results"] == []

    def test_hybrid_path_forwards_allergen_exclusions(self, monkeypatch):
        seen = {}

        class FakeHS:
            def search(self, query, top_k, **kw):
                seen.update(kw)
                return []

        monkeypatch.setitem(sys.modules, "retailgraph.graph.hybrid_search",
                            types.SimpleNamespace(HybridSearch=FakeHS))
        hybrid_search_node({"query": "protein bars", "entities": {
            "exclude_allergens": ["peanut"], "brand": "Kind", "min_price": 2}})
        assert seen == {"exclude_allergens": ["peanut"], "brand": "Kind", "min_price": 2}


# Empty LLM Cypher on a lookup must count a retry, or the graph loops until LangGraph's recursion limit.
def test_empty_llm_cypher_counts_a_retry():
    out = execute_query({"cypher_query": None, "cypher_valid": False, "cypher_retries": 0})
    assert out["cypher_retries"] == 1


# A lone zero price is still a real filter, so it must produce a template.
def test_zero_max_price_still_builds_template():
    assert _try_template({"max_price": 0}) is not None
