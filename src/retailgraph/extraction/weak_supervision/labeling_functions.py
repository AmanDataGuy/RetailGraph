"""
# labeling_functions.py - the 31 Snorkel labeling functions (LFs) for dietary tags and categories

- Each LF reads one row (catalog_content, price, image_link) and votes POSITIVE / NEGATIVE / ABSTAIN,
  or a CAT_* id / CAT_ABSTAIN. Absence of a keyword is never evidence: abstain instead of voting NEGATIVE.
- DIETARY_TAG_LFS and CATEGORY_LFS are consumed by label_model.py.
"""

from __future__ import annotations

import re

from snorkel.labeling import labeling_function

# Dietary tag labels
ABSTAIN   = -1
NEGATIVE  =  0
POSITIVE  =  1

# Category label ids (category LFs vote these; a separate LabelModel is trained for categories)
CAT_ABSTAIN     = -1
CAT_BEVERAGES   =  0
CAT_SNACKS      =  1
CAT_COFFEE_TEA  =  2
CAT_CONDIMENTS  =  3
CAT_GRAINS      =  4
CAT_BAKING      =  5
CAT_SPICES      =  6
CAT_SUPPLEMENTS =  7
CAT_NUTS        =  8
CAT_PERSONAL    =  9
CAT_PROTEIN_BAR = 10


# Lowercased full catalog_content; LFs search the whole text rather than parsing fields.
def _text(x) -> str:
    return str(x.catalog_content).lower()


# Organic: "organic" / "certified organic" / "usda organic".
@labeling_function()
def lf_organic_keyword(x):
    pattern = r'\b(organic|certified organic|usda organic)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# Kosher: "kosher" variants or the OU / OK certification marks.
@labeling_function()
def lf_kosher_keyword(x):
    pattern = r'\b(kosher|kosher certified|kosher pareve|kosher dairy|\bOU\b|\bOK\b)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# Gluten-free: "gluten free" or "gluten-free".
@labeling_function()
def lf_gluten_free_keyword(x):
    pattern = r'\bgluten[\s\-]free\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# Non-GMO: "non gmo" or "non-gmo".
@labeling_function()
def lf_non_gmo_keyword(x):
    pattern = r'\bnon[\s\-]gmo\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# Vegan: "vegan", "plant-based", "100% plant" (not "vegetarian", which allows dairy/eggs).
@labeling_function()
def lf_vegan_keyword(x):
    pattern = r'\b(vegan|plant[\s\-]based|100%\s*plant)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# Keto: "keto", "ketogenic" and friendly/certified variants.
@labeling_function()
def lf_keto_keyword(x):
    pattern = r'\b(keto|ketogenic|keto[\s\-]friendly|keto[\s\-]certified)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# Paleo: "paleo", "paleolithic" and friendly/diet variants.
@labeling_function()
def lf_paleo_keyword(x):
    pattern = r'\b(paleo|paleo[\s\-]friendly|paleo[\s\-]diet|paleolithic)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# Dairy-free: "dairy free", "lactose free" or "non-dairy".
@labeling_function()
def lf_dairy_free_keyword(x):
    pattern = r'\b(dairy[\s\-]free|lactose[\s\-]free|non[\s\-]dairy)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# Sugar-free: "sugar free", "no added sugar", "zero sugar", "unsweetened".
@labeling_function()
def lf_sugar_free_keyword(x):
    pattern = r'\b(sugar[\s\-]free|no added sugar|zero sugar|unsweetened)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# Nut-free: nut / tree-nut / peanut free (the schema has one tag for all three).
@labeling_function()
def lf_nut_free_keyword(x):
    pattern = r'\b(nut[\s\-]free|tree[\s\-]nut[\s\-]free|peanut[\s\-]free)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# Soy-free: "soy free" or "soy-free".
@labeling_function()
def lf_soy_free_keyword(x):
    pattern = r'\bsoy[\s\-]free\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# High-protein: only when the text says it explicitly.
@labeling_function()
def lf_high_protein_keyword(x):
    pattern = r'\b(high[\s\-]protein|protein[\s\-]rich|good source of protein|excellent source of protein)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# Category Beverages: juice, soda, waters, energy/sports drinks (coffee/tea has its own LF).
@labeling_function()
def lf_category_beverages(x):
    pattern = r'\b(juice|soda|sparkling water|mineral water|energy drink|sports drink|lemonade|kombucha|smoothie|coconut water)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_BEVERAGES
    return CAT_ABSTAIN


# Category Coffee & Tea: coffee, espresso, tea types, matcha, chai.
@labeling_function()
def lf_category_coffee_tea(x):
    pattern = r'\b(coffee|espresso|latte|cappuccino|tea|matcha|chai|herbal tea|green tea|black tea|oolong)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_COFFEE_TEA
    return CAT_ABSTAIN


# Category Snacks & Candy: chips, cookies, candy, jerky and similar (not the broad word "snack").
@labeling_function()
def lf_category_snacks(x):
    pattern = r'\b(chips|crackers|cookies|candy|chocolate|popcorn|pretzels|gummies|jerky|trail mix|granola)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_SNACKS
    return CAT_ABSTAIN


# Category Condiments & Sauces: sauces, dressings, vinegar, ketchup, mustard.
@labeling_function()
def lf_category_condiments(x):
    pattern = r'\b(sauce|salsa|ketchup|mustard|mayo|mayonnaise|dressing|vinegar|hot sauce|soy sauce|marinade|relish|chutney)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_CONDIMENTS
    return CAT_ABSTAIN


# Category Grains, Beans & Legumes: rice, lentils, beans, pasta, oats.
@labeling_function()
def lf_category_grains(x):
    pattern = r'\b(rice|lentils|chickpeas|black beans|kidney beans|pasta|quinoa|oats|barley|farro|couscous|millet)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_GRAINS
    return CAT_ABSTAIN


# Category Spices & Seasonings: common spices plus the McCormick brand.
@labeling_function()
def lf_category_spices(x):
    pattern = r'\b(spice|seasoning|pepper|cumin|turmeric|paprika|cinnamon|oregano|basil|thyme|garlic powder|onion powder|mccormick)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_SPICES
    return CAT_ABSTAIN


# Category Supplements & Health: vitamins, protein powder, probiotics, fish oil.
@labeling_function()
def lf_category_supplements(x):
    pattern = r'\b(supplement|vitamin|protein powder|whey|collagen|probiotics?|omega[\s\-]3|fish oil|multivitamin|zinc|magnesium|creatine)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_SUPPLEMENTS
    return CAT_ABSTAIN


# Category Nuts & Seeds: named nuts and seeds.
@labeling_function()
def lf_category_nuts(x):
    pattern = r'\b(almonds|cashews|walnuts|pecans|pistachios|macadamia|sunflower seeds|pumpkin seeds|chia seeds|flax seeds|hemp seeds)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_NUTS
    return CAT_ABSTAIN


# Category Baking & Cooking: flour, leaveners, extracts, cooking oils, mixes.
@labeling_function()
def lf_category_baking(x):
    pattern = r'\b(flour|baking soda|baking powder|yeast|cocoa powder|vanilla extract|vegetable oil|olive oil|coconut oil|shortening|bread mix|cake mix)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_BAKING
    return CAT_ABSTAIN


# Category Protein Bars & Snacks: bar types and bar brands (clif, rxbar, larabar...).
@labeling_function()
def lf_category_protein_bar(x):
    pattern = r'\b(protein bar|energy bar|granola bar|meal replacement bar|clif bar|rxbar|larabar|kind bar|quest bar|aloha bar)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_PROTEIN_BAR
    return CAT_ABSTAIN


# Category Spices from Hinglish names: masala, mirch, haldi, jeera...
@labeling_function()
def lf_hinglish_masala(x):
    pattern = r'\b(masala|mirch|haldi|jeera|dhania|hing|ajwain|methi|kali mirch|lal mirch|garam masala)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_SPICES
    return CAT_ABSTAIN


# Category Grains from Hinglish names: atta, dal, chawal, besan...
@labeling_function()
def lf_hinglish_grains(x):
    pattern = r'\b(atta|maida|besan|sooji|daliya|dal|chawal|chana|rajma|moong|urad)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_GRAINS
    return CAT_ABSTAIN


# NEGATIVE vegan when a vegan claim sits next to a dairy word (contradiction).
@labeling_function()
def lf_conflict_vegan_dairy(x):
    text = _text(x)
    has_vegan_claim = bool(re.search(r'\b(vegan|plant[\s\-]based)\b', text))
    has_dairy = bool(re.search(r'\b(milk|cheese|whey|butter|cream|yogurt|casein|lactose)\b', text))
    if has_vegan_claim and has_dairy:
        return NEGATIVE
    return ABSTAIN


# Weak organic signal: price over $50 plus "natural" / "pure" / "premium" / "artisan".
@labeling_function()
def lf_price_premium_organic(x):
    try:
        price = float(x.price)
    except (ValueError, TypeError):
        return ABSTAIN
    text = _text(x)
    is_expensive = price > 50
    has_natural = bool(re.search(r'\b(natural|pure|premium|artisan)\b', text))
    if is_expensive and has_natural:
        return POSITIVE
    return ABSTAIN


# Category Coffee & Tea when the image URL contains coffee/tea/espresso.
@labeling_function()
def lf_image_url_coffee(x):
    url = str(x.image_link).lower()
    if any(kw in url for kw in ['coffee', 'tea', 'espresso']):
        return CAT_COFFEE_TEA
    return CAT_ABSTAIN


# NEGATIVE vegan when a "contains ..." clause lists milk, egg or dairy.
@labeling_function()
def lf_allergen_implies_non_vegan(x):
    text = _text(x)
    dairy_allergen = bool(re.search(r'\bcontains[^.]*\b(milk|egg|dairy)\b', text))
    if dairy_allergen:
        return NEGATIVE
    return ABSTAIN


# Low-calorie: low/reduced calorie, "diet", or "light" + syrup/dressing/mayo/beer.
@labeling_function()
def lf_low_calorie_keyword(x):
    pattern = r'\b(low[\s\-]calorie|reduced[\s\-]calorie|diet\b|light\s+(?:syrup|dressing|mayo|beer))\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# Caffeine-free: "caffeine free", "decaf", "decaffeinated".
@labeling_function()
def lf_caffeine_free_keyword(x):
    pattern = r'\b(caffeine[\s\-]free|decaf|decaffeinated)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# Allergen-free: "allergen free", "free from", "top 8 allergen free".
@labeling_function()
def lf_allergen_free_keyword(x):
    pattern = r'\b(allergen[\s\-]free|free[\s\-]from|top[\s\-]8[\s\-]allergen[\s\-]free)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# LF registries used by label_model.py to build the two label matrices.
DIETARY_TAG_LFS = [
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
    lf_conflict_vegan_dairy,
    lf_price_premium_organic,
    lf_allergen_implies_non_vegan,
    lf_low_calorie_keyword,
    lf_caffeine_free_keyword,
    lf_allergen_free_keyword,
]

CATEGORY_LFS = [
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
    lf_hinglish_masala,
    lf_hinglish_grains,
    lf_image_url_coffee,
]