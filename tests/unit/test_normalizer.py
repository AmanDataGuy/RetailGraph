import pytest
from src.extraction.normalizer import (
    normalize_unit,
    normalize_dietary_tags,
    normalize_allergens,
    normalize_brand,
    normalize_category,
    extract_pack_size,
    clean_item_name,
    normalize_product,
)


# ── Unit Normalization Tests ──────────────────────────────────────────────────
class TestNormalizeUnit:

    def test_exact_alias_match(self):
        assert normalize_unit("Ounce") == "oz"
        assert normalize_unit("Count") == "ct"
        assert normalize_unit("Fl Oz") == "fl oz"
        assert normalize_unit("Pound") == "lb"

    def test_case_insensitive_match(self):
        assert normalize_unit("OUNCE") == "oz"
        assert normalize_unit("ounce") == "oz"
        assert normalize_unit("OZ") == "oz"

    def test_already_canonical_passthrough(self):
        for unit in ["oz", "fl oz", "lb", "ct", "g", "kg", "ml", "L"]:
            assert normalize_unit(unit) == unit

    def test_none_returns_none(self):
        assert normalize_unit(None) is None

    def test_string_none_returns_none(self):
        assert normalize_unit("None") is None

    def test_empty_string_returns_none(self):
        assert normalize_unit("") is None

    def test_whitespace_stripped(self):
        assert normalize_unit("  oz  ") == "oz"

    def test_unknown_unit_returns_none(self):
        assert normalize_unit("barrels") is None

    def test_pack_normalized_to_ct(self):
        assert normalize_unit("Pack") == "ct"
        assert normalize_unit("pack") == "ct"

    def test_count_variations(self):
        assert normalize_unit("COUNT") == "ct"
        assert normalize_unit("count") == "ct"
        assert normalize_unit("CT") == "ct"

    def test_fluid_ounce_variations(self):
        assert normalize_unit("Fluid Ounce") == "fl oz"
        assert normalize_unit("fluid oz") == "fl oz"
        assert normalize_unit("FL Oz") == "fl oz"
        assert normalize_unit("Fl. Oz") == "fl oz"


# ── Dietary Tag Normalization Tests ───────────────────────────────────────────
class TestNormalizeDietaryTags:

    def test_alias_normalization(self):
        result = normalize_dietary_tags(["gluten free"])
        assert result == ["gluten-free"]

    def test_multiple_tags(self):
        result = normalize_dietary_tags(["gluten free", "non gmo", "usda organic"])
        assert "gluten-free" in result
        assert "non-GMO" in result
        assert "organic" in result

    def test_duplicate_removal(self):
        result = normalize_dietary_tags(["organic", "organic", "vegan"])
        assert result.count("organic") == 1

    def test_none_returns_none(self):
        assert normalize_dietary_tags(None) is None

    def test_empty_list_returns_none(self):
        assert normalize_dietary_tags([]) is None

    def test_direct_canonical_passthrough(self):
        result = normalize_dietary_tags(["vegan", "kosher", "organic"])
        assert "vegan" in result
        assert "kosher" in result
        assert "organic" in result

    def test_case_insensitive(self):
        result = normalize_dietary_tags(["VEGAN", "Organic"])
        assert "vegan" in result
        assert "organic" in result

    def test_whitespace_stripped(self):
        result = normalize_dietary_tags(["  organic  "])
        assert "organic" in result

    def test_sugar_free_aliases(self):
        for alias in ["sugar free", "no sugar", "zero sugar", "no added sugar"]:
            result = normalize_dietary_tags([alias])
            assert result == ["sugar-free"]

    def test_keto_aliases(self):
        result = normalize_dietary_tags(["keto friendly"])
        assert result == ["keto"]

    def test_caffeine_free_aliases(self):
        result = normalize_dietary_tags(["decaf"])
        assert result == ["caffeine-free"]


# ── Allergen Normalization Tests ──────────────────────────────────────────────
class TestNormalizeAllergens:

    def test_canonical_passthrough(self):
        result = normalize_allergens(["milk", "egg", "peanut"])
        assert result == ["milk", "egg", "peanut"]

    def test_eggs_normalized_to_egg(self):
        result = normalize_allergens(["eggs"])
        assert result == ["egg"]

    def test_tree_nut_alias(self):
        result = normalize_allergens(["tree nuts"])
        assert "tree_nut" in result

    def test_duplicate_removal(self):
        result = normalize_allergens(["peanut", "peanut", "milk"])
        assert result.count("peanut") == 1

    def test_none_returns_none(self):
        assert normalize_allergens(None) is None

    def test_empty_list_returns_none(self):
        assert normalize_allergens([]) is None

    def test_all_valid_allergens_accepted(self):
        allergens = ["milk", "egg", "wheat", "soy", "peanut",
                     "fish", "shellfish", "sesame", "gluten"]
        result = normalize_allergens(allergens)
        assert len(result) == len(allergens)


# ── Brand Normalization Tests ─────────────────────────────────────────────────
class TestNormalizeBrand:

    def test_alias_match(self):
        assert normalize_brand("smuckers") == "Smucker's"
        assert normalize_brand("campbells") == "Campbell's"
        assert normalize_brand("annies") == "Annie's"

    def test_none_returns_none(self):
        assert normalize_brand(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_brand("") is None

    def test_whitespace_returns_none(self):
        assert normalize_brand("   ") is None

    def test_unknown_brand_returned_cleaned(self):
        result = normalize_brand("some random brand")
        assert result is not None
        assert isinstance(result, str)

    def test_whitespace_stripped(self):
        result = normalize_brand("  Starbucks  ")
        assert result is not None
        assert "Starbucks" in result


# ── Category Normalization Tests ──────────────────────────────────────────────
class TestNormalizeCategory:

    def test_exact_match(self):
        assert normalize_category("Beverages") == "Beverages"
        assert normalize_category("Coffee & Tea") == "Coffee & Tea"

    def test_none_returns_none(self):
        assert normalize_category(None) is None

    def test_empty_returns_none(self):
        assert normalize_category("") is None

    def test_fuzzy_match(self):
        result = normalize_category("Beverage")
        assert result == "Beverages"

    def test_unknown_category_returns_unknown(self):
        result = normalize_category("Underwater Basket Weaving")
        assert result == "Unknown"

    def test_all_valid_categories_exact(self):
        valid = [
            "Coffee & Tea", "Breakfast & Cereal", "Meat & Seafood",
            "Soups & Canned Goods", "Pasta & Noodles", "Bread & Bakery",
            "Protein Bars & Snacks", "Supplements & Health",
            "Grains, Beans & Legumes", "Oils & Vinegars", "Nuts & Seeds",
            "Personal Care & Beauty", "Spices & Seasonings",
            "Condiments & Sauces", "Baking & Cooking", "Snacks & Candy",
            "Beverages", "Non-Food", "Unknown",
        ]
        for cat in valid:
            assert normalize_category(cat) == cat


# ── Pack Size Extraction Tests ────────────────────────────────────────────────
class TestExtractPackSize:

    def test_pack_of_n(self):
        assert extract_pack_size("Smucker's Peanut Butter Pack of 12") == 12

    def test_n_pack(self):
        assert extract_pack_size("Coffee Pods 24-Pack") == 24

    def test_n_count(self):
        assert extract_pack_size("Vitamin C 60 Count") == 60

    def test_set_of_n(self):
        assert extract_pack_size("Set of 6 Jars") == 6

    def test_box_of_n(self):
        assert extract_pack_size("Box of 24 Tea Bags") == 24

    def test_case_of_n(self):
        assert extract_pack_size("Case of 12 Cans") == 12

    def test_none_returns_none(self):
        assert extract_pack_size(None) is None

    def test_no_pack_returns_none(self):
        assert extract_pack_size("Simple Organic Peanut Butter 16oz") is None

    def test_pack_of_1_returns_none(self):
        # pack_size of 1 is not meaningful
        assert extract_pack_size("Pack of 1") is None

    def test_extreme_value_returns_none(self):
        # 2560 is a resolution number not a pack size
        assert extract_pack_size("Image 2560px wide") is None

    def test_lowercase_pack(self):
        assert extract_pack_size("peanut butter pack of 6") == 6


# ── Item Name Cleaning Tests ──────────────────────────────────────────────────
class TestCleanItemName:

    def test_normal_name_unchanged(self):
        result = clean_item_name("Smucker's Natural Peanut Butter")
        assert result == "Smucker's Natural Peanut Butter"

    def test_whitespace_stripped(self):
        result = clean_item_name("  Peanut Butter  ")
        assert result == "Peanut Butter"

    def test_multiple_spaces_collapsed(self):
        result = clean_item_name("Peanut  Butter   16oz")
        assert result == "Peanut Butter 16oz"

    def test_trailing_punctuation_removed(self):
        result = clean_item_name("Peanut Butter,")
        assert result == "Peanut Butter"

    def test_none_returns_none(self):
        assert clean_item_name(None) is None

    def test_empty_returns_none(self):
        assert clean_item_name("") is None

    def test_too_short_returns_none(self):
        assert clean_item_name("A") is None

    def test_whitespace_only_returns_none(self):
        assert clean_item_name("   ") is None


# ── Full normalize_product Tests ──────────────────────────────────────────────
class TestNormalizeProduct:

    def test_basic_product_normalized(self):
        raw = {
            "product_id": "123",
            "item_name": "  Smucker's Peanut Butter  ",
            "price": 9.99,
            "quantity_unit": "Ounce",
            "dietary_tags": ["gluten free", "non gmo"],
        }
        result = normalize_product(raw)
        assert result["item_name"] == "Smucker's Peanut Butter"
        assert result["quantity_unit"] == "oz"
        assert "gluten-free" in result["dietary_tags"]
        assert "non-GMO" in result["dietary_tags"]

    def test_pack_size_extracted_from_item_name(self):
        raw = {
            "product_id": "456",
            "item_name": "Coffee Pods Pack of 12",
            "price": 15.99,
            "pack_size": None,
        }
        result = normalize_product(raw)
        assert result["pack_size"] == 12

    def test_explicit_pack_size_not_overridden(self):
        raw = {
            "product_id": "789",
            "item_name": "Coffee Pods Pack of 12",
            "price": 15.99,
            "pack_size": 6,  # explicitly set — should not be overridden
        }
        result = normalize_product(raw)
        assert result["pack_size"] == 6

    def test_string_none_unit_handled(self):
        raw = {
            "product_id": "321",
            "item_name": "Some Product",
            "price": 5.99,
            "quantity_unit": "None",
        }
        result = normalize_product(raw)
        assert result["quantity_unit"] is None

    def test_missing_fields_default_to_none(self):
        raw = {
            "product_id": "111",
            "item_name": "Minimal Product",
            "price": 4.99,
        }
        result = normalize_product(raw)
        assert result["brand"] is None
        assert result["category"] is None
        assert result["dietary_tags"] is None
        assert result["allergen_list"] is None

    def test_full_pipeline_produces_valid_entity(self):
        from src.extraction.schemas import ProductEntity
        raw = {
            "product_id": "999",
            "item_name": "  Organic Peanut Butter Pack of 6  ",
            "brand": "smuckers",
            "category": "Nuts & Seeds",
            "price": 36.0,
            "quantity_value": 16.0,
            "quantity_unit": "Ounce",
            "dietary_tags": ["usda organic", "gluten free"],
            "allergen_list": ["peanut"],
            "bullet_points": ["100% natural", "No preservatives",
                              "USDA Organic", "Gluten free", "Non-GMO"],
            "description": "Natural peanut butter made with roasted peanuts.",
            "extraction_confidence": 0.92,
        }
        cleaned = normalize_product(raw)
        entity = ProductEntity(**cleaned)
        entity.assign_content_tier()
        entity.compute_quality_score()

        assert entity.item_name == "Organic Peanut Butter Pack of 6"
        assert entity.brand == "Smucker's"
        assert entity.quantity_unit == "oz"
        assert "organic" in entity.dietary_tags
        assert "gluten-free" in entity.dietary_tags
        assert entity.pack_size == 6
        assert entity.unit_price == 6.0
        assert entity.is_organic is True
        assert entity.quality_score >= 70