import pytest
from src.extraction.schemas import (
    ContentTier,
    ImageQuality,
    PackagingType,
    ProductEntity,
    VisualAttributes,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def minimal_product():
    """Bare minimum valid product — only required fields."""
    return {
        "product_id": "12345",
        "item_name": "Test Product",
        "price": 9.99,
    }


@pytest.fixture
def rich_product():
    """Fully populated product — all fields present."""
    return {
        "product_id": "99999",
        "item_name": "Smucker's Natural Chunky Peanut Butter 16oz Pack of 12",
        "brand": "Smucker's",
        "category": "Nuts & Seeds",
        "quantity_value": 16.0,
        "quantity_unit": "oz",
        "pack_size": 12,
        "price": 51.06,
        "dietary_tags": ["organic", "kosher"],
        "allergen_list": ["peanut"],
        "description": "Natural peanut butter made with roasted peanuts.",
        "bullet_points": [
            "Made with roasted peanuts",
            "No hydrogenated oils",
            "Kosher certified",
            "Natural ingredients",
            "16oz jar",
        ],
        "image_url": "https://example.com/image.jpg",
        "visual": {
            "packaging_type": "jar",
            "packaging_color": "brown",
            "has_brand_logo": True,
            "image_quality": "high",
        },
    }


# ── Core Identity Tests ───────────────────────────────────────────────────────
class TestCoreIdentity:

    def test_minimal_product_passes(self, minimal_product):
        entity = ProductEntity(**minimal_product)
        assert entity.product_id == "12345"
        assert entity.item_name == "Test Product"
        assert entity.price == 9.99

    def test_missing_product_id_fails(self, minimal_product):
        del minimal_product["product_id"]
        with pytest.raises(Exception):
            ProductEntity(**minimal_product)

    def test_missing_item_name_fails(self, minimal_product):
        del minimal_product["item_name"]
        with pytest.raises(Exception):
            ProductEntity(**minimal_product)

    def test_missing_price_fails(self, minimal_product):
        del minimal_product["price"]
        with pytest.raises(Exception):
            ProductEntity(**minimal_product)

    def test_empty_product_id_fails(self, minimal_product):
        minimal_product["product_id"] = "   "
        with pytest.raises(Exception):
            ProductEntity(**minimal_product)

    def test_item_name_whitespace_stripped(self, minimal_product):
        minimal_product["item_name"] = "  Test Product  "
        entity = ProductEntity(**minimal_product)
        assert entity.item_name == "Test Product"

    def test_item_name_too_short_fails(self, minimal_product):
        minimal_product["item_name"] = "A"
        with pytest.raises(Exception):
            ProductEntity(**minimal_product)

    def test_item_name_too_long_fails(self, minimal_product):
        minimal_product["item_name"] = "A" * 501
        with pytest.raises(Exception):
            ProductEntity(**minimal_product)

    def test_brand_none_accepted(self, minimal_product):
        entity = ProductEntity(**minimal_product)
        assert entity.brand is None

    def test_brand_whitespace_returns_none(self, minimal_product):
        minimal_product["brand"] = "   "
        entity = ProductEntity(**minimal_product)
        assert entity.brand is None


# ── Price Tests ───────────────────────────────────────────────────────────────
class TestPrice:

    def test_zero_price_accepted(self, minimal_product):
        minimal_product["price"] = 0.0
        entity = ProductEntity(**minimal_product)
        assert entity.price == 0.0

    def test_very_low_price_accepted(self, minimal_product):
        minimal_product["price"] = 0.13
        entity = ProductEntity(**minimal_product)
        assert entity.price == 0.13

    def test_high_price_accepted(self, minimal_product):
        minimal_product["price"] = 2796.0
        entity = ProductEntity(**minimal_product)
        assert entity.price == 2796.0

    def test_negative_price_fails(self, minimal_product):
        minimal_product["price"] = -1.0
        with pytest.raises(Exception):
            ProductEntity(**minimal_product)


# ── Category Tests ────────────────────────────────────────────────────────────
class TestCategory:

    def test_valid_category_accepted(self, minimal_product):
        minimal_product["category"] = "Beverages"
        entity = ProductEntity(**minimal_product)
        assert entity.category == "Beverages"

    def test_all_valid_categories_accepted(self, minimal_product):
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
            minimal_product["category"] = cat
            entity = ProductEntity(**minimal_product)
            assert entity.category == cat

    def test_invalid_category_fails(self, minimal_product):
        minimal_product["category"] = "Drinks"
        with pytest.raises(Exception):
            ProductEntity(**minimal_product)

    def test_none_category_accepted(self, minimal_product):
        entity = ProductEntity(**minimal_product)
        assert entity.category is None


# ── Unit Tests ────────────────────────────────────────────────────────────────
class TestUnit:

    def test_all_valid_units_accepted(self, minimal_product):
        for unit in ["oz", "fl oz", "lb", "ct", "g", "kg", "ml", "L"]:
            minimal_product["quantity_unit"] = unit
            entity = ProductEntity(**minimal_product)
            assert entity.quantity_unit == unit

    def test_invalid_unit_fails(self, minimal_product):
        minimal_product["quantity_unit"] = "kilos"
        with pytest.raises(Exception):
            ProductEntity(**minimal_product)

    def test_none_unit_accepted(self, minimal_product):
        entity = ProductEntity(**minimal_product)
        assert entity.quantity_unit is None


# ── Dietary Tag Tests ─────────────────────────────────────────────────────────
class TestDietaryTags:

    def test_valid_tags_accepted(self, minimal_product):
        minimal_product["dietary_tags"] = ["organic", "vegan", "gluten-free"]
        entity = ProductEntity(**minimal_product)
        assert "organic" in entity.dietary_tags

    def test_invalid_tag_fails(self, minimal_product):
        minimal_product["dietary_tags"] = ["healthy"]
        with pytest.raises(Exception):
            ProductEntity(**minimal_product)

    def test_duplicate_tags_removed(self, minimal_product):
        minimal_product["dietary_tags"] = ["organic", "organic", "vegan"]
        entity = ProductEntity(**minimal_product)
        assert entity.dietary_tags.count("organic") == 1

    def test_empty_list_returns_none(self, minimal_product):
        minimal_product["dietary_tags"] = []
        entity = ProductEntity(**minimal_product)
        assert entity.dietary_tags is None

    def test_tags_lowercased(self, minimal_product):
    # Schema does NOT lowercase — normalizer does that before schema sees it
    # Passing un-normalized tags directly to schema should fail
        minimal_product["dietary_tags"] = ["Organic", "VEGAN"]
        with pytest.raises(Exception):
            ProductEntity(**minimal_product)


# ── Allergen Tests ────────────────────────────────────────────────────────────
class TestAllergens:

    def test_valid_allergens_accepted(self, minimal_product):
        minimal_product["allergen_list"] = ["peanut", "milk", "egg"]
        entity = ProductEntity(**minimal_product)
        assert entity.allergen_list == ["peanut", "milk", "egg"]

    def test_invalid_allergen_fails(self, minimal_product):
        minimal_product["allergen_list"] = ["nuts"]
        with pytest.raises(Exception):
            ProductEntity(**minimal_product)

    def test_duplicate_allergens_removed(self, minimal_product):
        minimal_product["allergen_list"] = ["peanut", "peanut"]
        entity = ProductEntity(**minimal_product)
        assert entity.allergen_list.count("peanut") == 1


# ── Derived Fields Tests ──────────────────────────────────────────────────────
class TestDerivedFields:

    def test_unit_price_computed(self, minimal_product):
        minimal_product["price"] = 12.0
        minimal_product["pack_size"] = 6
        entity = ProductEntity(**minimal_product)
        assert entity.unit_price == 2.0

    def test_unit_price_not_computed_for_pack_of_1(self, minimal_product):
        minimal_product["price"] = 12.0
        minimal_product["pack_size"] = 1
        entity = ProductEntity(**minimal_product)
        assert entity.unit_price is None

    def test_is_organic_derived_true(self, minimal_product):
        minimal_product["dietary_tags"] = ["organic", "vegan"]
        entity = ProductEntity(**minimal_product)
        assert entity.is_organic is True

    def test_is_organic_derived_false(self, minimal_product):
        minimal_product["dietary_tags"] = ["vegan", "kosher"]
        entity = ProductEntity(**minimal_product)
        assert entity.is_organic is False

    def test_is_kosher_derived_true(self, minimal_product):
        minimal_product["dietary_tags"] = ["kosher"]
        entity = ProductEntity(**minimal_product)
        assert entity.is_kosher is True

    def test_is_organic_none_when_no_tags(self, minimal_product):
        entity = ProductEntity(**minimal_product)
        assert entity.is_organic is None


# ── Content Tier Tests ────────────────────────────────────────────────────────
class TestContentTier:

    def test_rich_tier_assigned(self, minimal_product):
        minimal_product["bullet_points"] = ["b1", "b2", "b3", "b4", "b5"]
        minimal_product["description"] = "A great product."
        entity = ProductEntity(**minimal_product)
        entity.assign_content_tier()
        assert entity.content_tier == ContentTier.RICH

    def test_medium_tier_assigned(self, minimal_product):
        minimal_product["bullet_points"] = ["b1", "b2"]
        entity = ProductEntity(**minimal_product)
        entity.assign_content_tier()
        assert entity.content_tier == ContentTier.MEDIUM

    def test_bare_tier_assigned(self, minimal_product):
        entity = ProductEntity(**minimal_product)
        entity.assign_content_tier()
        assert entity.content_tier == ContentTier.BARE

    def test_five_bullets_no_description_is_medium(self, minimal_product):
        minimal_product["bullet_points"] = ["b1", "b2", "b3", "b4", "b5"]
        entity = ProductEntity(**minimal_product)
        entity.assign_content_tier()
        assert entity.content_tier == ContentTier.MEDIUM


# ── Quality Score Tests ───────────────────────────────────────────────────────
class TestQualityScore:

    def test_minimal_product_score(self, minimal_product):
        entity = ProductEntity(**minimal_product)
        entity.compute_quality_score()
        assert entity.quality_score == 20  # item_name (15) + price (5)

    def test_rich_product_score(self, rich_product):
        entity = ProductEntity(**rich_product)
        entity.compute_quality_score()
        assert entity.quality_score >= 70

    def test_score_capped_at_100(self, rich_product):
        entity = ProductEntity(**rich_product)
        entity.compute_quality_score()
        assert entity.quality_score <= 100

    def test_score_never_negative(self, minimal_product):
        entity = ProductEntity(**minimal_product)
        entity.compute_quality_score()
        assert entity.quality_score >= 0


# ── VisualAttributes Tests ────────────────────────────────────────────────────
class TestVisualAttributes:

    def test_visual_attributes_accepted(self, minimal_product):
        minimal_product["visual"] = {
            "packaging_type": "bottle",
            "packaging_color": "red",
            "has_brand_logo": True,
            "image_quality": "high",
        }
        entity = ProductEntity(**minimal_product)
        assert entity.visual.packaging_type == PackagingType.BOTTLE
        assert entity.visual.packaging_color == "red"

    def test_visual_none_accepted(self, minimal_product):
        entity = ProductEntity(**minimal_product)
        assert entity.visual is None

    def test_packaging_color_lowercased(self, minimal_product):
        minimal_product["visual"] = {
            "packaging_color": "RED",
            "image_quality": "high",
        }
        entity = ProductEntity(**minimal_product)
        assert entity.visual.packaging_color == "red"

    def test_invalid_packaging_type_fails(self, minimal_product):
        minimal_product["visual"] = {
            "packaging_type": "plastic_wrap",
            "image_quality": "high",
        }
        with pytest.raises(Exception):
            ProductEntity(**minimal_product)

    def test_invalid_image_quality_fails(self, minimal_product):
        minimal_product["visual"] = {
            "image_quality": "ultra",
        }
        with pytest.raises(Exception):
            ProductEntity(**minimal_product)