# src/extraction/weak_supervision/labeling_functions.py
"""
Labeling functions for RetailGraph weak supervision pipeline.

Each LF looks at one product row (pandas Series with fields parsed
from catalog_content) and returns:
    POSITIVE ( 1) — I am confident this label applies
    NEGATIVE ( 0) — I am confident this label does NOT apply
    ABSTAIN  (-1) — I have no signal, skip me

Rules:
    - Never return NEGATIVE unless you have positive evidence against.
      Absence of a keyword is NOT evidence — use ABSTAIN.
    - Keep each LF focused on ONE label only.
    - Every LF must be deterministic — same input, same output always.
"""

from __future__ import annotations

import re
import yaml
from pathlib import Path
from snorkel.labeling import labeling_function

# ── Label constants ───────────────────────────────────────────────────────────
# Dietary tag labels
ABSTAIN   = -1
NEGATIVE  =  0
POSITIVE  =  1

# Category labels — each category gets its own integer
# We run a separate LabelModel per label type (tag vs category)
# For category LFs we use these constants instead:
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

# ── Load domain adapter ───────────────────────────────────────────────────────
# We load retail.yaml once at module level so every LF can use
# the same brand/allergen/dietary dictionaries without re-reading
# the file on every function call.
_ADAPTER_PATH = Path("src/domain_adapter/retail.yaml")

def _load_adapter() -> dict:
    with open(_ADAPTER_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)

_ADAPTER = _load_adapter()
_BRAND_ALIASES: dict  = _ADAPTER.get("brand_aliases", {})
_DIETARY_ALIASES: dict = _ADAPTER.get("dietary_tag_aliases", {})
_ALLERGEN_ALIASES: dict = _ADAPTER.get("allergen_aliases", {})
_CATEGORY_KEYWORDS: dict = _ADAPTER.get("category_keywords", {})

# ── Helper — get full text from a row ────────────────────────────────────────
# catalog_content contains everything: item_name, bullets, description.
# Rather than re-parsing in every LF, we just search the whole string.
# This is fast enough at 75k rows and keeps LFs simple.

def _text(x) -> str:
    """Return lowercased full catalog_content for a row."""
    return str(x.catalog_content).lower()

def _name(x) -> str:
    """Return just the item_name portion, lowercased."""
    match = re.search(r'item name:\s*(.+?)(?:\n|$)', str(x.catalog_content), re.IGNORECASE)
    return match.group(1).lower() if match else ""


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 1 — Keyword & Regex LFs (dietary tags)
# These are the strongest signal LFs — they fire when a specific word
# or phrase appears directly in the product text.
# ══════════════════════════════════════════════════════════════════════════════

# ── LF 1 — Organic keyword ───────────────────────────────────────────────────
# "organic" appears 10,431 times in our dataset (EDA Cell 5).
# \b ensures we don't match "inorganic". re.IGNORECASE handles all caps.
# We ABSTAIN rather than voting NEGATIVE because a product without
# "organic" in the text might still be organic — just not labelled that way.

@labeling_function()
def lf_organic_keyword(x):
    pattern = r'\b(organic|certified organic|usda organic)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# ── LF 2 — Kosher keyword ────────────────────────────────────────────────────
# Kosher is the most common dietary tag in our dataset (13,175 — EDA Cell 5).
# "kosher certified", "kosher pareve", "kosher dairy" are all valid forms.
# OU, OK, KOF-K are common kosher certification symbols on packaging.

@labeling_function()
def lf_kosher_keyword(x):
    pattern = r'\b(kosher|kosher certified|kosher pareve|kosher dairy|\bOU\b|\bOK\b)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# ── LF 3 — Gluten-free keyword ───────────────────────────────────────────────
# EDA showed "gluten free" (9,922) + "gluten-free" (6,229) = 16,151 total.
# Both spellings must be caught. Also catches "certified gluten free" and
# "gluten free certified" (word order varies on real labels).

@labeling_function()
def lf_gluten_free_keyword(x):
    pattern = r'\bgluten[\s\-]free\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# ── LF 4 — Non-GMO keyword ───────────────────────────────────────────────────
# "non-gmo" appears 9,460 times (EDA). Also catches the NGP butterfly logo
# text ("non gmo project verified") which is very common on Amazon listings.

@labeling_function()
def lf_non_gmo_keyword(x):
    pattern = r'\bnon[\s\-]gmo\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# ── LF 5 — Vegan keyword ─────────────────────────────────────────────────────
# "vegan" = 8,418 (EDA). "plant-based" and "100% plant" are aliases
# from retail.yaml. We don't catch "vegetarian" here — that's a different
# and weaker signal (vegetarian products can contain dairy/eggs).

@labeling_function()
def lf_vegan_keyword(x):
    pattern = r'\b(vegan|plant[\s\-]based|100%\s*plant)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# ── LF 6 — Keto keyword ──────────────────────────────────────────────────────
# "keto" = 3,297 (EDA). "ketogenic" is the formal name.
# "keto friendly" and "keto certified" are also valid.
# Note: keto products are often also low-carb — but we only label keto here.

@labeling_function()
def lf_keto_keyword(x):
    pattern = r'\b(keto|ketogenic|keto[\s\-]friendly|keto[\s\-]certified)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# ── LF 7 — Paleo keyword ─────────────────────────────────────────────────────
# "paleo" = 1,472 (EDA). Less common than keto.
# "paleo friendly" and "paleo diet" are the main variants seen in listings.

@labeling_function()
def lf_paleo_keyword(x):
    pattern = r'\b(paleo|paleo[\s\-]friendly|paleo[\s\-]diet|paleolithic)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# ── LF 8 — Dairy-free keyword ────────────────────────────────────────────────
# "dairy free" (1,399) + "dairy-free" (883) = 2,282 total (EDA).
# "lactose free" is a different but strongly correlated signal —
# almost all lactose-free products are also dairy-free in this dataset.

@labeling_function()
def lf_dairy_free_keyword(x):
    pattern = r'\b(dairy[\s\-]free|lactose[\s\-]free|non[\s\-]dairy)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# ── LF 9 — Sugar-free keyword ────────────────────────────────────────────────
# "sugar free" (2,029) + "sugar-free" (1,309) = 3,338 total (EDA).
# "no added sugar" and "zero sugar" are strong aliases for the same concept.
# "unsweetened" is a weaker signal — some unsweetened products still have
# natural sugars — so we include it but it adds less certainty.

@labeling_function()
def lf_sugar_free_keyword(x):
    pattern = r'\b(sugar[\s\-]free|no added sugar|zero sugar|unsweetened)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# ── LF 10 — Nut-free keyword ─────────────────────────────────────────────────
# "nut free" (1,086) + "nut-free" (623) = 1,709 total (EDA).
# "tree nut free" is a more specific version — means no almonds, cashews etc.
# "peanut free" is separate (peanuts are legumes, not tree nuts) but
# we include it here because our schema treats them the same tag.

@labeling_function()
def lf_nut_free_keyword(x):
    pattern = r'\b(nut[\s\-]free|tree[\s\-]nut[\s\-]free|peanut[\s\-]free)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# ── LF 11 — Soy-free keyword ─────────────────────────────────────────────────
# "soy free" (713) + "soy-free" (346) = 1,059 total (EDA).
# Smaller signal than gluten/dairy but still worth catching directly.

@labeling_function()
def lf_soy_free_keyword(x):
    pattern = r'\bsoy[\s\-]free\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# ── LF 12 — High protein keyword ─────────────────────────────────────────────
# "high protein" doesn't appear directly in our EDA counts but it's
# one of the 15 controlled dietary tags in retail.yaml.
# Common in supplements, protein bars, Greek yogurt listings.
# We use a conservative threshold — only fire when explicitly stated.

@labeling_function()
def lf_high_protein_keyword(x):
    pattern = r'\b(high[\s\-]protein|protein[\s\-]rich|good source of protein|excellent source of protein)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 2 — Category LFs + Hinglish + Conflict detection
# ══════════════════════════════════════════════════════════════════════════════

# ── LF 13 — Beverages category ───────────────────────────────────────────────
# Largest category (19,752 — EDA Cell 11). Broad keywords.
# We exclude coffee/tea keywords here — those have their own LF.
# "juice", "soda", "water", "drink" are the cleanest signals.

@labeling_function()
def lf_category_beverages(x):
    pattern = r'\b(juice|soda|sparkling water|mineral water|energy drink|sports drink|lemonade|kombucha|smoothie|coconut water)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_BEVERAGES
    return CAT_ABSTAIN


# ── LF 14 — Coffee & Tea category ────────────────────────────────────────────
# 12,374 products (EDA). The most specific category — very clean signal.
# "espresso", "matcha", "herbal tea" are unambiguous.

@labeling_function()
def lf_category_coffee_tea(x):
    pattern = r'\b(coffee|espresso|latte|cappuccino|tea|matcha|chai|herbal tea|green tea|black tea|oolong)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_COFFEE_TEA
    return CAT_ABSTAIN


# ── LF 15 — Snacks & Candy category ─────────────────────────────────────────
# 17,355 products (EDA). Wide variety — chips, cookies, candy, crackers.
# We avoid generic words like "snack" alone — too broad.

@labeling_function()
def lf_category_snacks(x):
    pattern = r'\b(chips|crackers|cookies|candy|chocolate|popcorn|pretzels|gummies|jerky|trail mix|granola)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_SNACKS
    return CAT_ABSTAIN


# ── LF 16 — Condiments & Sauces category ─────────────────────────────────────
# 5,591 products (EDA). Very clean signals — sauce, dressing, vinegar.

@labeling_function()
def lf_category_condiments(x):
    pattern = r'\b(sauce|salsa|ketchup|mustard|mayo|mayonnaise|dressing|vinegar|hot sauce|soy sauce|marinade|relish|chutney)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_CONDIMENTS
    return CAT_ABSTAIN


# ── LF 17 — Grains, Beans & Legumes category ─────────────────────────────────
# 3,012 products (EDA). Rice, lentils, beans, pasta.

@labeling_function()
def lf_category_grains(x):
    pattern = r'\b(rice|lentils|chickpeas|black beans|kidney beans|pasta|quinoa|oats|barley|farro|couscous|millet)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_GRAINS
    return CAT_ABSTAIN


# ── LF 18 — Spices & Seasonings category ─────────────────────────────────────
# 2,368 products (EDA). Pepper, cumin, turmeric etc.
# McCormick is the top brand (621 products from EDA Cell 7) — use it.

@labeling_function()
def lf_category_spices(x):
    pattern = r'\b(spice|seasoning|pepper|cumin|turmeric|paprika|cinnamon|oregano|basil|thyme|garlic powder|onion powder|mccormick)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_SPICES
    return CAT_ABSTAIN


# ── LF 19 — Supplements & Health category ────────────────────────────────────
# 1,972 products (EDA) — likely undercounted. Protein powder, vitamins.

@labeling_function()
def lf_category_supplements(x):
    pattern = r'\b(supplement|vitamin|protein powder|whey|collagen|probiotics?|omega[\s\-]3|fish oil|multivitamin|zinc|magnesium|creatine)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_SUPPLEMENTS
    return CAT_ABSTAIN


# ── LF 20 — Nuts & Seeds category ────────────────────────────────────────────
# 1,209 products (EDA). Almonds, cashews, sunflower seeds etc.

@labeling_function()
def lf_category_nuts(x):
    pattern = r'\b(almonds|cashews|walnuts|pecans|pistachios|macadamia|sunflower seeds|pumpkin seeds|chia seeds|flax seeds|hemp seeds)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_NUTS
    return CAT_ABSTAIN


# ── LF 21 — Baking & Cooking category ────────────────────────────────────────
# 2,536 products (EDA). Flour, sugar, baking soda, oils.

@labeling_function()
def lf_category_baking(x):
    pattern = r'\b(flour|baking soda|baking powder|yeast|cocoa powder|vanilla extract|vegetable oil|olive oil|coconut oil|shortening|bread mix|cake mix)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_BAKING
    return CAT_ABSTAIN


# ── LF 22 — Protein Bars category ────────────────────────────────────────────
# Only 4 products caught initially (EDA) — fixed with better keywords.
# Brand names are the strongest signal here (clif, rxbar, larabar etc.)

@labeling_function()
def lf_category_protein_bar(x):
    pattern = r'\b(protein bar|energy bar|granola bar|meal replacement bar|clif bar|rxbar|larabar|kind bar|quest bar|aloha bar)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_PROTEIN_BAR
    return CAT_ABSTAIN


# ── LF 23 — Hinglish: Spices & Masala ────────────────────────────────────────
# Indian grocery products use transliterated Hindi names.
# From EDA we know this dataset has US Amazon products — but Rani brand
# (425 products, EDA Cell 7) sells Indian spices with Hinglish names.
# "masala", "mirch", "haldi" etc. are unambiguous spice signals.

@labeling_function()
def lf_hinglish_masala(x):
    pattern = r'\b(masala|mirch|haldi|jeera|dhania|hing|ajwain|methi|kali mirch|lal mirch|garam masala)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_SPICES
    return CAT_ABSTAIN


# ── LF 24 — Hinglish: Grains & Staples ───────────────────────────────────────
# "atta" (wheat flour), "dal" (lentils), "chawal" (rice), "besan" (chickpea flour).
# These are unambiguous grain/legume signals when they appear in item names.

@labeling_function()
def lf_hinglish_grains(x):
    pattern = r'\b(atta|maida|besan|sooji|daliya|dal|chawal|chana|rajma|moong|urad)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return CAT_GRAINS
    return CAT_ABSTAIN


# ── LF 25 — Conflict detection: vegan + dairy ────────────────────────────────
# If a product claims to be vegan BUT also mentions milk/dairy/cheese/whey,
# that's a contradiction. We vote NEGATIVE on vegan for these products.
# This is the only LF that returns NEGATIVE — because we have positive
# evidence against the label (confirmed dairy ingredient present).

@labeling_function()
def lf_conflict_vegan_dairy(x):
    text = _text(x)
    has_vegan_claim = bool(re.search(r'\b(vegan|plant[\s\-]based)\b', text))
    has_dairy = bool(re.search(r'\b(milk|cheese|whey|butter|cream|yogurt|casein|lactose)\b', text))
    if has_vegan_claim and has_dairy:
        return NEGATIVE   # contradiction — definitely not vegan
    return ABSTAIN


# ── LF 26 — Price-based: premium organic signal ───────────────────────────────
# From EDA: mean price $23.6, 75th percentile $28.6.
# Products priced above $50 with "natural" or "pure" in the name
# are very likely premium/organic. Weak signal — use low weight.
# This LF uses price column directly (not catalog_content).

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


# ── LF 27 — Image filename: category signal ───────────────────────────────────
# image_link URLs contain the product ASIN which doesn't give category info,
# but sometimes the URL path contains category hints.
# This is a weak LF — most Amazon image URLs don't contain text keywords.
# We keep it because it adds independent signal from a different feature.

@labeling_function()
def lf_image_url_coffee(x):
    url = str(x.image_link).lower()
    if any(kw in url for kw in ['coffee', 'tea', 'espresso']):
        return CAT_COFFEE_TEA
    return CAT_ABSTAIN


# ── LF 28 — Allergen implies non-vegan ───────────────────────────────────────
# If allergen_list (or text) contains milk or egg — product is not vegan.
# This is positive evidence for NEGATIVE vegan label.

@labeling_function()
def lf_allergen_implies_non_vegan(x):
    text = _text(x)
    dairy_allergen = bool(re.search(r'\bcontains[^.]*\b(milk|egg|dairy)\b', text))
    if dairy_allergen:
        return NEGATIVE
    return ABSTAIN


# ── LF 29 — Low calorie keyword ──────────────────────────────────────────────
# "low calorie" and "light" are common on diet products.
# "light" alone is too ambiguous (could be light roast coffee, light color)
# so we require it with a supporting context word.

@labeling_function()
def lf_low_calorie_keyword(x):
    pattern = r'\b(low[\s\-]calorie|reduced[\s\-]calorie|diet\b|light\s+(?:syrup|dressing|mayo|beer))\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# ── LF 30 — Caffeine-free keyword ────────────────────────────────────────────
# "caffeine free", "decaf", "decaffeinated". Strong, unambiguous signals.
# Mostly fires on coffee and tea products.

@labeling_function()
def lf_caffeine_free_keyword(x):
    pattern = r'\b(caffeine[\s\-]free|decaf|decaffeinated)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# ── LF 31 — Allergen-free keyword ────────────────────────────────────────────
# "allergen free" is a catch-all tag for products free from all major allergens.
# "free from" is a UK/international phrasing also seen in imports.
# "top 8 allergen free" refers to FDA's top 8 allergens.

@labeling_function()
def lf_allergen_free_keyword(x):
    pattern = r'\b(allergen[\s\-]free|free[\s\-]from|top[\s\-]8[\s\-]allergen[\s\-]free)\b'
    if re.search(pattern, _text(x), re.IGNORECASE):
        return POSITIVE
    return ABSTAIN


# ── LF registries ─────────────────────────────────────────────────────────────
# Two separate lists — one for dietary tag LFs, one for category LFs.
# Snorkel needs these to build the label matrix.
# Import these lists in label_model.py.

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