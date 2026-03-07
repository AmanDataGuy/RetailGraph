# tests/unit/test_labeling_functions.py
"""
Unit tests for all labeling functions.

Strategy:
    - Each LF gets a POSITIVE case (should fire) and an ABSTAIN case (should not fire).
    - Conflict LFs also get a NEGATIVE case.
    - We use pandas Series to simulate real rows — same as what Snorkel passes in.
    - We never test with real dataset rows — hand-crafted examples only,
      so tests are deterministic and don't depend on data files.
"""

import pandas as pd
import pytest

from src.extraction.weak_supervision.labeling_functions import (
    # Constants
    ABSTAIN, POSITIVE, NEGATIVE,
    CAT_ABSTAIN, CAT_BEVERAGES, CAT_COFFEE_TEA, CAT_SNACKS,
    CAT_CONDIMENTS, CAT_GRAINS, CAT_SPICES, CAT_SUPPLEMENTS,
    CAT_NUTS, CAT_BAKING, CAT_PROTEIN_BAR,
    # Dietary tag LFs
    lf_organic_keyword,
    lf_kosher_keyword,
    lf_gluten_free_keyword,
    lf_non_gmo_keyword,
    lf_vegan_keyword,
    lf_keto_keyword,
    lf_paleo_keyword,
    lf_dairy_free_keyword,
    lf_sugar_free_keyword,
    lf_nut_free_keyword,
    lf_soy_free_keyword,
    lf_high_protein_keyword,
    lf_low_calorie_keyword,
    lf_caffeine_free_keyword,
    lf_allergen_free_keyword,
    # Category LFs
    lf_category_beverages,
    lf_category_coffee_tea,
    lf_category_snacks,
    lf_category_condiments,
    lf_category_grains,
    lf_category_spices,
    lf_category_supplements,
    lf_category_nuts,
    lf_category_baking,
    lf_category_protein_bar,
    # Hinglish LFs
    lf_hinglish_masala,
    lf_hinglish_grains,
    # Conflict LFs
    lf_conflict_vegan_dairy,
    lf_allergen_implies_non_vegan,
    # Price + image LFs
    lf_price_premium_organic,
    lf_image_url_coffee,
)


# ── Helper ────────────────────────────────────────────────────────────────────
def make_row(catalog_content: str, price: float = 9.99, image_link: str = "") -> pd.Series:
    """
    Simulate a single dataset row as a pandas Series.
    This is exactly what Snorkel passes into each @labeling_function.
    """
    return pd.Series({
        "catalog_content": catalog_content,
        "price": price,
        "image_link": image_link,
    })


# ── Dietary Tag LF Tests ──────────────────────────────────────────────────────

class TestOrganicKeyword:
    def test_fires_on_organic(self):
        row = make_row("Item Name: Organic Valley Whole Milk\nBullet Point 1: USDA Organic certified")
        assert lf_organic_keyword(row) == POSITIVE

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Regular Whole Milk 1 Gallon")
        assert lf_organic_keyword(row) == ABSTAIN

    def test_does_not_match_inorganic(self):
        # "inorganic" should NOT trigger the organic label
        row = make_row("Item Name: Inorganic Chemistry Supplement")
        assert lf_organic_keyword(row) == ABSTAIN

    def test_case_insensitive(self):
        row = make_row("Item Name: CERTIFIED ORGANIC Green Tea")
        assert lf_organic_keyword(row) == POSITIVE


class TestKosherKeyword:
    def test_fires_on_kosher(self):
        row = make_row("Item Name: Hebrew National Beef Franks\nBullet Point 1: Kosher certified")
        assert lf_kosher_keyword(row) == POSITIVE

    def test_fires_on_kosher_pareve(self):
        row = make_row("Item Name: Dark Chocolate Bar\nBullet Point 1: Kosher Pareve OU certified")
        assert lf_kosher_keyword(row) == POSITIVE

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Plain White Rice 2lb")
        assert lf_kosher_keyword(row) == ABSTAIN


class TestGlutenFreeKeyword:
    def test_fires_on_gluten_free_hyphen(self):
        row = make_row("Item Name: Bob's Red Mill Gluten-Free Oats")
        assert lf_gluten_free_keyword(row) == POSITIVE

    def test_fires_on_gluten_free_space(self):
        row = make_row("Item Name: Ancient Grain Bread\nBullet Point 1: Gluten Free certified")
        assert lf_gluten_free_keyword(row) == POSITIVE

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Whole Wheat Pasta 16oz")
        assert lf_gluten_free_keyword(row) == ABSTAIN


class TestNonGmoKeyword:
    def test_fires_on_non_gmo_hyphen(self):
        row = make_row("Item Name: Organic Soybeans\nBullet Point 1: Non-GMO Project Verified")
        assert lf_non_gmo_keyword(row) == POSITIVE

    def test_fires_on_non_gmo_space(self):
        row = make_row("Item Name: Canola Oil\nBullet Point 1: non gmo certified")
        assert lf_non_gmo_keyword(row) == POSITIVE

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Corn Starch 16oz")
        assert lf_non_gmo_keyword(row) == ABSTAIN


class TestVeganKeyword:
    def test_fires_on_vegan(self):
        row = make_row("Item Name: Follow Your Heart Vegan Cheese")
        assert lf_vegan_keyword(row) == POSITIVE

    def test_fires_on_plant_based(self):
        row = make_row("Item Name: Impossible Burger Plant-Based Patties")
        assert lf_vegan_keyword(row) == POSITIVE

    def test_abstains_on_vegetarian(self):
        # vegetarian is NOT the same as vegan — should abstain
        row = make_row("Item Name: Vegetarian Chili No Beans")
        assert lf_vegan_keyword(row) == ABSTAIN

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Chicken Breast 2lb")
        assert lf_vegan_keyword(row) == ABSTAIN


class TestKetoKeyword:
    def test_fires_on_keto(self):
        row = make_row("Item Name: Perfect Keto Bars Chocolate Chip")
        assert lf_keto_keyword(row) == POSITIVE

    def test_fires_on_ketogenic(self):
        row = make_row("Item Name: MCT Oil\nBullet Point 1: Supports ketogenic diet")
        assert lf_keto_keyword(row) == POSITIVE

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: White Bread Loaf 20oz")
        assert lf_keto_keyword(row) == ABSTAIN


class TestPaleoKeyword:
    def test_fires_on_paleo(self):
        row = make_row("Item Name: Epic Provisions Chicken Sriracha Bars Paleo Friendly")
        assert lf_paleo_keyword(row) == POSITIVE

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Whole Grain Cereal 18oz")
        assert lf_paleo_keyword(row) == ABSTAIN


class TestDairyFreeKeyword:
    def test_fires_on_dairy_free_hyphen(self):
        row = make_row("Item Name: So Delicious Dairy-Free Coconut Milk")
        assert lf_dairy_free_keyword(row) == POSITIVE

    def test_fires_on_lactose_free(self):
        row = make_row("Item Name: Lactaid Lactose Free Milk")
        assert lf_dairy_free_keyword(row) == POSITIVE

    def test_fires_on_non_dairy(self):
        row = make_row("Item Name: Coffee Mate Non-Dairy Creamer")
        assert lf_dairy_free_keyword(row) == POSITIVE

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Whole Milk Mozzarella Cheese")
        assert lf_dairy_free_keyword(row) == ABSTAIN


class TestSugarFreeKeyword:
    def test_fires_on_sugar_free(self):
        row = make_row("Item Name: Jello Sugar Free Gelatin Dessert")
        assert lf_sugar_free_keyword(row) == POSITIVE

    def test_fires_on_no_added_sugar(self):
        row = make_row("Item Name: Apple Juice\nBullet Point 1: No added sugar, just real fruit")
        assert lf_sugar_free_keyword(row) == POSITIVE

    def test_fires_on_zero_sugar(self):
        row = make_row("Item Name: Zevia Zero Sugar Cola")
        assert lf_sugar_free_keyword(row) == POSITIVE

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Honey Roasted Peanuts 16oz")
        assert lf_sugar_free_keyword(row) == ABSTAIN


class TestNutFreeKeyword:
    def test_fires_on_nut_free(self):
        row = make_row("Item Name: SunButter Sunflower Seed Butter\nBullet Point 1: Nut-free facility")
        assert lf_nut_free_keyword(row) == POSITIVE

    def test_fires_on_peanut_free(self):
        row = make_row("Item Name: School Safe Snack Bars Peanut Free")
        assert lf_nut_free_keyword(row) == POSITIVE

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Mixed Nuts Variety Pack 1lb")
        assert lf_nut_free_keyword(row) == ABSTAIN


class TestSoyFreeKeyword:
    def test_fires_on_soy_free(self):
        row = make_row("Item Name: Against The Grain Gourmet Pizza\nBullet Point 1: Soy-free and gluten-free")
        assert lf_soy_free_keyword(row) == POSITIVE

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Edamame Frozen Soybeans 10oz")
        assert lf_soy_free_keyword(row) == ABSTAIN


class TestHighProteinKeyword:
    def test_fires_on_high_protein(self):
        row = make_row("Item Name: Optimum Nutrition Gold Standard Whey\nBullet Point 1: High-protein formula 24g per serving")
        assert lf_high_protein_keyword(row) == POSITIVE

    def test_fires_on_good_source(self):
        row = make_row("Item Name: Greek Yogurt Plain\nBullet Point 1: Good source of protein")
        assert lf_high_protein_keyword(row) == POSITIVE

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: White Rice 5lb Bag")
        assert lf_high_protein_keyword(row) == ABSTAIN


class TestLowCalorieKeyword:
    def test_fires_on_low_calorie(self):
        row = make_row("Item Name: Walden Farms Low-Calorie Pancake Syrup")
        assert lf_low_calorie_keyword(row) == POSITIVE

    def test_fires_on_diet(self):
        row = make_row("Item Name: Diet Coke 12 Pack Cans")
        assert lf_low_calorie_keyword(row) == POSITIVE

    def test_abstains_on_light_roast(self):
        # "light" alone should not trigger — "light roast" is a coffee term
        row = make_row("Item Name: Starbucks Veranda Light Roast Coffee")
        assert lf_low_calorie_keyword(row) == ABSTAIN

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Peanut Butter Cups 12oz")
        assert lf_low_calorie_keyword(row) == ABSTAIN


class TestCaffeineFreeKeyword:
    def test_fires_on_caffeine_free(self):
        row = make_row("Item Name: Celestial Seasonings Caffeine-Free Herbal Tea")
        assert lf_caffeine_free_keyword(row) == POSITIVE

    def test_fires_on_decaf(self):
        row = make_row("Item Name: Folgers Decaf Classic Roast Ground Coffee")
        assert lf_caffeine_free_keyword(row) == POSITIVE

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Starbucks Pike Place Roast Coffee")
        assert lf_caffeine_free_keyword(row) == ABSTAIN


class TestAllergenFreeKeyword:
    def test_fires_on_allergen_free(self):
        row = make_row("Item Name: Enjoy Life Soft Baked Cookies\nBullet Point 1: Allergen-free, top 8 free")
        assert lf_allergen_free_keyword(row) == POSITIVE

    def test_fires_on_free_from(self):
        row = make_row("Item Name: Free From Gluten Pasta\nBullet Point 1: Free-from all major allergens")
        assert lf_allergen_free_keyword(row) == POSITIVE

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Peanut Butter Smooth 16oz")
        assert lf_allergen_free_keyword(row) == ABSTAIN


# ── Category LF Tests ─────────────────────────────────────────────────────────

class TestCategoryBeverages:
    def test_fires_on_juice(self):
        row = make_row("Item Name: Tropicana Orange Juice 52oz")
        assert lf_category_beverages(row) == CAT_BEVERAGES

    def test_fires_on_kombucha(self):
        row = make_row("Item Name: GT's Synergy Kombucha Gingerade 16oz")
        assert lf_category_beverages(row) == CAT_BEVERAGES

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Organic Valley Whole Milk 1 Gallon")
        assert lf_category_beverages(row) == CAT_ABSTAIN


class TestCategoryCoffeeTea:
    def test_fires_on_coffee(self):
        row = make_row("Item Name: Starbucks Pike Place Ground Coffee 12oz")
        assert lf_category_coffee_tea(row) == CAT_COFFEE_TEA

    def test_fires_on_matcha(self):
        row = make_row("Item Name: Jade Leaf Organic Matcha Green Tea Powder")
        assert lf_category_coffee_tea(row) == CAT_COFFEE_TEA

    def test_fires_on_chai(self):
        row = make_row("Item Name: Tazo Chai Tea Latte Concentrate")
        assert lf_category_coffee_tea(row) == CAT_COFFEE_TEA

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Apple Cider Vinegar 16oz")
        assert lf_category_coffee_tea(row) == CAT_ABSTAIN


class TestCategorySnacks:
    def test_fires_on_chips(self):
        row = make_row("Item Name: Lay's Classic Potato Chips 8oz")
        assert lf_category_snacks(row) == CAT_SNACKS

    def test_fires_on_chocolate(self):
        row = make_row("Item Name: Ghirardelli Dark Chocolate Bar 3.5oz")
        assert lf_category_snacks(row) == CAT_SNACKS

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Brown Rice Long Grain 5lb")
        assert lf_category_snacks(row) == CAT_ABSTAIN


class TestCategoryCondiments:
    def test_fires_on_salsa(self):
        row = make_row("Item Name: La Victoria Green Taco Sauce Mild 12oz")
        assert lf_category_condiments(row) == CAT_CONDIMENTS

    def test_fires_on_mustard(self):
        row = make_row("Item Name: French's Classic Yellow Mustard 30oz")
        assert lf_category_condiments(row) == CAT_CONDIMENTS

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: White Rice 5lb")
        assert lf_category_condiments(row) == CAT_ABSTAIN


class TestCategoryGrains:
    def test_fires_on_rice(self):
        row = make_row("Item Name: Lundberg Family Farms Organic Brown Rice 2lb")
        assert lf_category_grains(row) == CAT_GRAINS

    def test_fires_on_lentils(self):
        row = make_row("Item Name: Bob's Red Mill Green Lentils 27oz")
        assert lf_category_grains(row) == CAT_GRAINS

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Olive Oil Extra Virgin 16oz")
        assert lf_category_grains(row) == CAT_ABSTAIN


class TestCategorySpices:
    def test_fires_on_paprika(self):
        row = make_row("Item Name: McCormick Smoked Paprika 1.75oz")
        assert lf_category_spices(row) == CAT_SPICES

    def test_fires_on_mccormick_brand(self):
        row = make_row("Item Name: McCormick Garlic Powder 3.12oz")
        assert lf_category_spices(row) == CAT_SPICES

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Almond Milk Unsweetened 32oz")
        assert lf_category_spices(row) == CAT_ABSTAIN


class TestCategorySupplements:
    def test_fires_on_protein_powder(self):
        row = make_row("Item Name: Optimum Nutrition Gold Standard 100% Whey Protein Powder")
        assert lf_category_supplements(row) == CAT_SUPPLEMENTS

    def test_fires_on_probiotic(self):
        row = make_row("Item Name: Garden of Life Raw Probiotics Women 85 Billion")
        assert lf_category_supplements(row) == CAT_SUPPLEMENTS

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Granola Bars Variety Pack 12 Count")
        assert lf_category_supplements(row) == CAT_ABSTAIN


class TestCategoryNuts:
    def test_fires_on_almonds(self):
        row = make_row("Item Name: Blue Diamond Almonds Whole Natural 16oz")
        assert lf_category_nuts(row) == CAT_NUTS

    def test_fires_on_chia_seeds(self):
        row = make_row("Item Name: Navitas Organics Chia Seeds 8oz")
        assert lf_category_nuts(row) == CAT_NUTS

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Chocolate Chip Cookies 13oz")
        assert lf_category_nuts(row) == CAT_ABSTAIN


class TestCategoryBaking:
    def test_fires_on_flour(self):
        row = make_row("Item Name: King Arthur Unbleached All-Purpose Flour 5lb")
        assert lf_category_baking(row) == CAT_BAKING

    def test_fires_on_baking_soda(self):
        row = make_row("Item Name: Arm and Hammer Pure Baking Soda 8oz")
        assert lf_category_baking(row) == CAT_BAKING

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Chicken Noodle Soup 18.6oz")
        assert lf_category_baking(row) == CAT_ABSTAIN


class TestCategoryProteinBar:
    def test_fires_on_protein_bar(self):
        row = make_row("Item Name: Quest Nutrition Protein Bar Chocolate Chip Cookie Dough")
        assert lf_category_protein_bar(row) == CAT_PROTEIN_BAR

    def test_fires_on_clif_bar(self):
        row = make_row("Item Name: CLIF BAR Energy Bars Chocolate Chip 12 Count")
        assert lf_category_protein_bar(row) == CAT_PROTEIN_BAR

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Organic Whole Milk Greek Yogurt")
        assert lf_category_protein_bar(row) == CAT_ABSTAIN


# ── Hinglish LF Tests ─────────────────────────────────────────────────────────

class TestHinglishMasala:
    def test_fires_on_masala(self):
        row = make_row("Item Name: Rani Brand Garam Masala Indian 11-Spice Blend 3oz")
        assert lf_hinglish_masala(row) == CAT_SPICES

    def test_fires_on_haldi(self):
        row = make_row("Item Name: Badia Haldi Turmeric Ground 16oz")
        assert lf_hinglish_masala(row) == CAT_SPICES

    def test_fires_on_jeera(self):
        row = make_row("Item Name: Frontier Co-op Jeera Cumin Seed Whole 1lb")
        assert lf_hinglish_masala(row) == CAT_SPICES

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Organic Apple Cider Vinegar 16oz")
        assert lf_hinglish_masala(row) == CAT_ABSTAIN


class TestHinglishGrains:
    def test_fires_on_atta(self):
        row = make_row("Item Name: Deep Whole Wheat Atta Flour 20lb")
        assert lf_hinglish_grains(row) == CAT_GRAINS

    def test_fires_on_dal(self):
        row = make_row("Item Name: Swad Moong Dal Yellow Split 4lb")
        assert lf_hinglish_grains(row) == CAT_GRAINS

    def test_fires_on_besan(self):
        row = make_row("Item Name: Laxmi Besan Chickpea Flour 2lb")
        assert lf_hinglish_grains(row) == CAT_GRAINS

    def test_abstains_on_no_signal(self):
        row = make_row("Item Name: Starbucks House Blend Coffee 12oz")
        assert lf_hinglish_grains(row) == CAT_ABSTAIN


# ── Conflict Detection LF Tests ───────────────────────────────────────────────

class TestConflictVeganDairy:
    def test_fires_negative_on_contradiction(self):
        # Claims vegan but lists milk as ingredient — contradiction
        row = make_row("Item Name: Vegan Cheese Slice\nBullet Point 1: Contains milk and casein")
        assert lf_conflict_vegan_dairy(row) == NEGATIVE

    def test_abstains_on_clean_vegan(self):
        # Vegan claim with no dairy — no contradiction, don't vote
        row = make_row("Item Name: Follow Your Heart Vegan Parmesan\nBullet Point 1: 100% plant-based")
        assert lf_conflict_vegan_dairy(row) == ABSTAIN

    def test_abstains_on_dairy_no_vegan_claim(self):
        # Has dairy but never claimed vegan — no conflict
        row = make_row("Item Name: Sharp Cheddar Cheese 8oz")
        assert lf_conflict_vegan_dairy(row) == ABSTAIN


class TestAllergenImpliesNonVegan:
    def test_fires_negative_on_contains_milk(self):
        row = make_row("Item Name: Milk Chocolate Bar\nBullet Point 1: Contains milk, soy, wheat")
        assert lf_allergen_implies_non_vegan(row) == NEGATIVE

    def test_abstains_when_no_contains_statement(self):
        # Mentions milk in context other than "contains" — abstain
        row = make_row("Item Name: Oat Milk Latte\nBullet Point 1: dairy-free alternative to milk")
        assert lf_allergen_implies_non_vegan(row) == ABSTAIN

    def test_abstains_on_no_allergen(self):
        row = make_row("Item Name: Roasted Almonds Unsalted 16oz")
        assert lf_allergen_implies_non_vegan(row) == ABSTAIN


# ── Price + Image LF Tests ────────────────────────────────────────────────────

class TestPricePremiumOrganic:
    def test_fires_on_expensive_natural(self):
        row = make_row(
            "Item Name: Pure Natural Manuka Honey Premium Grade",
            price=75.0
        )
        assert lf_price_premium_organic(row) == POSITIVE

    def test_abstains_on_cheap_natural(self):
        # Natural but not expensive — weak signal, abstain
        row = make_row("Item Name: Natural Peanut Butter 16oz", price=6.99)
        assert lf_price_premium_organic(row) == ABSTAIN

    def test_abstains_on_expensive_no_signal(self):
        # Expensive but no natural/pure/premium keyword
        row = make_row("Item Name: Fancy Chocolate Assortment Gift Box", price=80.0)
        assert lf_price_premium_organic(row) == ABSTAIN

    def test_abstains_on_bad_price(self):
        row = make_row("Item Name: Pure Vanilla Extract", price=float("nan"))
        assert lf_price_premium_organic(row) == ABSTAIN


class TestImageUrlCoffee:
    def test_fires_on_coffee_in_url(self):
        row = make_row(
            "Item Name: Dark Roast Coffee Beans",
            image_link="https://images.amazon.com/coffee-beans-dark-roast.jpg"
        )
        assert lf_image_url_coffee(row) == CAT_COFFEE_TEA

    def test_abstains_on_no_keyword_in_url(self):
        row = make_row(
            "Item Name: Organic Almonds",
            image_link="https://images.amazon.com/I/51mo8htwTH.jpg"
        )
        assert lf_image_url_coffee(row) == CAT_ABSTAIN