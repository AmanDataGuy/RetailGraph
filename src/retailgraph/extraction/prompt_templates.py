"""
# prompt_templates.py - every prompt used to make an LLM emit ProductEntity JSON

- Holds the system prompt, few-shot examples, controlled vocabularies and chat-message builders.
- Used by training/generate_pairs.py, generate_synthetic.py and generate_visual_pairs.py (retry messages).
"""
from __future__ import annotations

# Exact values the model must choose from.
ALLOWED_CATEGORIES = [
    "Beverages",
    "Coffee & Tea",
    "Snacks & Candy",
    "Condiments & Sauces",
    "Grains, Beans & Legumes",
    "Baking & Cooking",
    "Spices & Seasonings",
    "Supplements & Health",
    "Nuts & Seeds",
    "Personal Care & Beauty",
    "Protein Bars & Snacks",
]

ALLOWED_DIETARY_TAGS = [
    "organic",
    "cruelty-free",
    "kosher",
    "gluten-free",
    "non-GMO",
    "vegan",
    "keto",
    "paleo",
    "dairy-free",
    "sugar-free",
    "nut-free",
    "soy-free",
    "high-protein",
    "low-calorie",
    "caffeine-free",
    "allergen-free",
]

CANONICAL_UNITS = [
    "oz",
    "fl oz",
    "lb",
    "g",
    "kg",
    "ml",
    "l",
    "ct",
    "pack",
]

# Deliberately verbose: explicit rules mean fewer validation failures.

SYSTEM_PROMPT = """You are a product data extraction model for a grocery e-commerce platform.

Your job is to read a raw product catalog listing and extract structured information into a valid JSON object.

## OUTPUT RULES — FOLLOW EXACTLY

1. Return ONLY a valid JSON object. No markdown, no backticks, no explanation, no preamble.
2. Never add fields that are not in the schema below.
3. Never omit required fields (item_name, price, quantity_value, quantity_unit, category, extraction_confidence).
4. Use ONLY the exact values from the controlled vocabulary lists below.
5. If a field cannot be determined from the text, use the default value specified.

## OUTPUT SCHEMA

{
  "item_name": string,           // Clean product name WITHOUT size/quantity. Max 120 chars.
  "price": float,                // Product price as a positive number. Required.
  "quantity_value": float,       // Numeric amount (e.g. 16.0 for 16oz). Default: 1.0
  "quantity_unit": string,       // MUST be one of the canonical units below.
  "category": string,            // MUST be one of the allowed categories below.
  "dietary_tags": [string],      // List of applicable tags. Only from allowed list. Empty [] if none.
  "allergen_list": [string],     // Allergens mentioned. Lowercase. Empty [] if none.
  "extraction_confidence": float // Your confidence in this extraction. Float between 0.0 and 1.0.
}

## CANONICAL UNITS — use EXACTLY these strings
oz, fl oz, lb, g, kg, ml, l, ct, pack

Unit conversion rules:
  "Ounce" / "ounces" / "OZ"  → oz
  "Fluid Ounce" / "fl. oz."  → fl oz
  "Count" / "count" / "Ct"   → ct
  "Pound" / "pounds" / "LB"  → lb
  "Gram" / "grams"           → g
  "Milliliter" / "ml"        → ml
  "Liter" / "liters"         → l
  If unit is unclear or missing → ct

## ALLOWED CATEGORIES — use EXACTLY these strings
Beverages, Coffee & Tea, Snacks & Candy, Condiments & Sauces,
Grains, Beans & Legumes, Baking & Cooking, Spices & Seasonings,
Supplements & Health, Nuts & Seeds, Personal Care & Beauty,
Protein Bars & Snacks

## ALLOWED DIETARY TAGS — only use these exact strings
organic, kosher, gluten-free, non-GMO, vegan, keto, paleo,
dairy-free, sugar-free, nut-free, soy-free, high-protein,
low-calorie, caffeine-free, allergen-free, cruelty-free

## CONFIDENCE CALIBRATION
Set extraction_confidence based on how much information is available:
  0.90–0.97 → Rich listing: item name + 3+ bullet points + clear price + clear unit
  0.75–0.89 → Medium listing: item name + 1-2 bullets, some fields inferrable
  0.55–0.74 → Sparse listing: item name only, many fields inferred or defaulted
  0.40–0.54 → Very sparse or ambiguous: only partial name, unclear category
  Never output 1.0 — no extraction is perfectly certain.
  Never output below 0.40 for a product that exists.

## ITEM NAME RULES
  - Remove size/quantity from name: "Smucker's Peanut Butter 16oz" → "Smucker's Natural Peanut Butter"
  - Remove pack info: "McCormick Garlic Powder 3.12oz (Pack of 6)" → "McCormick Garlic Powder"
  - Keep brand name: Always preserve the brand
  - Max 120 characters
"""

# Worked examples covering edge cases, Hinglish names and realistic confidence scores.

FEW_SHOT_EXAMPLES = [
    # 1: Rich English listing — spice, kosher, non-GMO
    {
        "input": """Item Name: McCormick Garlic Powder 3.12oz
Bullet Point 1: Pure garlic powder with no added MSG
Bullet Point 2: Kosher certified
Bullet Point 3: Non-GMO Project Verified
Bullet Point 4: Perfect for marinades, sauces and rubs
Value: 3.12
Unit: Ounce
Price: 4.99""",
        "output": """{
  "item_name": "McCormick Garlic Powder",
  "price": 4.99,
  "quantity_value": 3.12,
  "quantity_unit": "oz",
  "category": "Spices & Seasonings",
  "dietary_tags": ["kosher", "non-GMO"],
  "allergen_list": [],
  "extraction_confidence": 0.95
}""",
    },

    # 2: Beverage with multiple dietary tags
    {
        "input": """Item Name: Celestial Seasonings Caffeine-Free Herbal Tea Variety Pack 40 Count
Bullet Point 1: Naturally caffeine free herbal tea
Bullet Point 2: Non-GMO verified, no artificial flavors
Bullet Point 3: Kosher certified
Bullet Point 4: 40 individually wrapped tea bags
Value: 40
Unit: Count
Price: 8.49""",
        "output": """{
  "item_name": "Celestial Seasonings Caffeine-Free Herbal Tea Variety Pack",
  "price": 8.49,
  "quantity_value": 40.0,
  "quantity_unit": "ct",
  "category": "Coffee & Tea",
  "dietary_tags": ["caffeine-free", "non-GMO", "kosher"],
  "allergen_list": [],
  "extraction_confidence": 0.93
}""",
    },

    # 3: Hinglish product — Indian spice blend
    {
        "input": """Item Name: Rani Garam Masala Indian 11-Spice Blend 3oz
Bullet Point 1: Premium quality Indian spice blend
Bullet Point 2: All natural, no preservatives, no artificial colors
Bullet Point 3: Gluten free, vegan
Value: 3
Unit: Ounce
Price: 8.99""",
        "output": """{
  "item_name": "Rani Garam Masala Indian 11-Spice Blend",
  "price": 8.99,
  "quantity_value": 3.0,
  "quantity_unit": "oz",
  "category": "Spices & Seasonings",
  "dietary_tags": ["gluten-free", "vegan"],
  "allergen_list": [],
  "extraction_confidence": 0.91
}""",
    },

    # 4: Protein bar with allergen
    {
        "input": """Item Name: RXBAR Chocolate Sea Salt Protein Bar 12 Count
Bullet Point 1: 12g protein, no added sugar
Bullet Point 2: Gluten free, dairy free
Bullet Point 3: Contains: Egg Whites, Almonds, Cashews, Dates
Value: 12
Unit: Count
Price: 21.99""",
        "output": """{
  "item_name": "RXBAR Chocolate Sea Salt Protein Bar",
  "price": 21.99,
  "quantity_value": 12.0,
  "quantity_unit": "ct",
  "category": "Protein Bars & Snacks",
  "dietary_tags": ["gluten-free", "dairy-free", "high-protein", "sugar-free"],
  "allergen_list": ["egg", "almond", "cashew"],
  "extraction_confidence": 0.94
}""",
    },

    # 5: Sparse listing — bare item name only
    {
        "input": """Item Name: Goya Black Beans 15.5oz
Value: 15.5
Unit: Ounce
Price: 1.29""",
        "output": """{
  "item_name": "Goya Black Beans",
  "price": 1.29,
  "quantity_value": 15.5,
  "quantity_unit": "oz",
  "category": "Grains, Beans & Legumes",
  "dietary_tags": [],
  "allergen_list": [],
  "extraction_confidence": 0.72
}""",
    },

    # 6: Hinglish grains — atta (wheat flour)
    {
        "input": """Item Name: Aashirvaad Whole Wheat Atta 10 Lb
Bullet Point 1: Made from 100% whole wheat
Bullet Point 2: No maida, no preservatives
Value: 10
Unit: Pound
Price: 12.99""",
        "output": """{
  "item_name": "Aashirvaad Whole Wheat Atta",
  "price": 12.99,
  "quantity_value": 10.0,
  "quantity_unit": "lb",
  "category": "Grains, Beans & Legumes",
  "dietary_tags": [],
  "allergen_list": ["wheat"],
  "extraction_confidence": 0.88
}""",
    },

    # 7: Supplement with multiple tags
    {
        "input": """Item Name: Garden of Life Organic Protein Powder Vanilla 22oz
Bullet Point 1: USDA Organic, non-GMO project verified
Bullet Point 2: Vegan, gluten free, soy free
Bullet Point 3: 22g protein per serving
Value: 22
Unit: Ounce
Price: 38.49""",
        "output": """{
  "item_name": "Garden of Life Organic Protein Powder Vanilla",
  "price": 38.49,
  "quantity_value": 22.0,
  "quantity_unit": "oz",
  "category": "Supplements & Health",
  "dietary_tags": ["organic", "non-GMO", "vegan", "gluten-free", "soy-free", "high-protein"],
  "allergen_list": [],
  "extraction_confidence": 0.96
}""",
    },

    # 8: Condiment with allergen
    {
        "input": """Item Name: Heinz Tomato Ketchup 32oz
Bullet Point 1: Made from red ripe tomatoes
Bullet Point 2: No high fructose corn syrup
Value: 32
Unit: Ounce
Price: 4.79""",
        "output": """{
  "item_name": "Heinz Tomato Ketchup",
  "price": 4.79,
  "quantity_value": 32.0,
  "quantity_unit": "oz",
  "category": "Condiments & Sauces",
  "dietary_tags": [],
  "allergen_list": [],
  "extraction_confidence": 0.89
}""",
    },
]


# The system prompt, trimmed.
def build_system_prompt() -> str:
    return SYSTEM_PROMPT.strip()


# User turn for one product; appends the price when the listing does not already have one.
def build_user_prompt(catalog_content: str, price: float | None = None) -> str:
    content = catalog_content.strip()
    if price is not None and "Price:" not in content:
        content = f"{content}\nPrice: {price}"

    return f"""Extract product information from the following grocery product listing.
Return ONLY a valid JSON object matching the schema. No explanation, no markdown, no backticks.

Product listing:
{content}"""


# Few-shot examples as alternating user/assistant chat messages.
def build_few_shot_messages() -> list[dict]:
    messages = []
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": build_user_prompt(example["input"])})
        messages.append({"role": "assistant", "content": example["output"].strip()})
    return messages


# Full chat for one product: system prompt, optional few-shot examples, then the product.
def build_full_messages(
    catalog_content: str,
    price: float | None = None,
    include_few_shot: bool = True,
) -> list[dict]:
    messages = [{"role": "system", "content": build_system_prompt()}]
    if include_few_shot:
        messages.extend(build_few_shot_messages())
    messages.append({"role": "user", "content": build_user_prompt(catalog_content, price)})
    return messages


# Copy of the chat plus the failed reply and the error, asking the model to fix it.
def build_retry_messages(
    original_messages: list[dict],
    failed_output: str,
    error_description: str,
) -> list[dict]:
    return list(original_messages) + [
        {"role": "assistant", "content": failed_output},
        {
            "role": "user",
            "content": (
                f"That output failed validation with this error:\n{error_description}\n\n"
                f"Fix the issue and return ONLY the corrected JSON object. "
                f"No explanation, no markdown, no backticks."
            ),
        },
    ]
