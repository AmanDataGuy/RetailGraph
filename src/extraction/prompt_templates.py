"""
Prompt templates for ProductEntity extraction.

PURPOSE:
    This file owns every prompt used to instruct LLMs (Groq/GPT-4o-mini/Qwen2-VL)
    to extract structured ProductEntity JSON from raw catalog text.

    Used in three places:
        1. training/generate_pairs.py   — generates 3,000 seed training pairs
        2. training/generate_synthetic.py — generates synthetic products
        3. scripts/run_extraction.py    — runs Qwen2-VL on 72k products

WHY STRICT PROMPTS MATTER:
    The fine-tuned Qwen2-VL model learns EXACTLY what we show it.
    If the prompt is loose and the LLM sometimes returns markdown,
    sometimes returns extra explanation, or sometimes uses wrong field
    names — the model learns all those bad habits too.
    A strict, consistent prompt produces consistent JSON every time.

PROMPT DESIGN PRINCIPLES:
    1. System prompt defines the contract — what the model IS and what it MUST do
    2. User prompt provides the raw input and repeats the output contract
    3. Few-shot examples demonstrate correct behavior for edge cases
    4. Multi-language examples (English + Hinglish) handle our dataset's diversity
    5. Confidence calibration examples teach realistic confidence scores (not 1.0)
"""

from __future__ import annotations

from typing import Optional


# ── Controlled vocabulary (must match retail.yaml + schemas.py) ───────────────
# These are the EXACT values the model must choose from.
# Any deviation fails Pydantic validation in validator.py.

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

# ── System prompt ─────────────────────────────────────────────────────────────
# The system prompt is sent once per session and defines the model's role,
# strict output rules, and controlled vocabulary.
# It is intentionally verbose — the more explicit the rules, the fewer
# validation failures we get from generate_pairs.py.

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
low-calorie, caffeine-free, allergen-free

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

# ── Few-shot examples ─────────────────────────────────────────────────────────
# These demonstrate correct behavior for different product types and edge cases.
# Multi-language coverage ensures the model handles Hinglish products.
# Confidence calibration examples teach realistic scores.

FEW_SHOT_EXAMPLES = [
    # ── Example 1: Rich English listing — spice, kosher, non-GMO ─────────────
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

    # ── Example 2: Beverage with multiple dietary tags ────────────────────────
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

    # ── Example 3: Hinglish product — Indian spice blend ──────────────────────
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

    # ── Example 4: Protein bar with allergen ─────────────────────────────────
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

    # ── Example 5: Sparse listing — bare item name only ───────────────────────
    # Tests confidence calibration for low-information products
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

    # ── Example 6: Hinglish grains — atta (wheat flour) ──────────────────────
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

    # ── Example 7: Supplement with multiple tags ──────────────────────────────
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

    # ── Example 8: Condiment with allergen ───────────────────────────────────
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


# ── Template builder functions ────────────────────────────────────────────────

def build_system_prompt() -> str:
    """
    Return the system prompt string.

    Used as the 'system' message in every LLM API call.
    Defines the model's role, schema, and strict output rules.

    Returns:
        Full system prompt string.
    """
    return SYSTEM_PROMPT.strip()


def build_user_prompt(catalog_content: str, price: Optional[float] = None) -> str:
    """
    Build the user-turn prompt for a single product.

    Takes the raw catalog_content from train.csv and formats it
    into a prompt that asks for ProductEntity JSON extraction.

    Args:
        catalog_content: raw text from the catalog_content column
        price:           product price from the price column (may be None)

    Returns:
        User prompt string ready to send to the LLM.
    """
    # Append price to catalog_content if available and not already present.
    # Some products have price embedded in the text, some don't.
    content = catalog_content.strip()
    if price is not None and "Price:" not in content:
        content = f"{content}\nPrice: {price}"

    return f"""Extract product information from the following grocery product listing.
Return ONLY a valid JSON object matching the schema. No explanation, no markdown, no backticks.

Product listing:
{content}"""


def build_few_shot_messages() -> list[dict]:
    """
    Build a list of few-shot message dicts in OpenAI/Groq chat format.

    The few-shot examples are injected as alternating user/assistant
    messages before the actual product prompt. This is the most
    reliable way to demonstrate output format to instruct-tuned models.

    Returns:
        List of {'role': ..., 'content': ...} dicts.
        Alternates: user (example input) → assistant (example output)
    """
    messages = []
    for example in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": build_user_prompt(example["input"])
        })
        messages.append({
            "role": "assistant",
            "content": example["output"].strip()
        })
    return messages


def build_full_messages(
    catalog_content: str,
    price: Optional[float] = None,
    include_few_shot: bool = True,
) -> list[dict]:
    """
    Build the complete message list for one product extraction API call.

    Structure:
        [system prompt]
        [few-shot example 1 user]
        [few-shot example 1 assistant]
        ...
        [few-shot example N user]
        [few-shot example N assistant]
        [actual product user prompt]  ← this is what the model responds to

    The system prompt + few-shot examples together constrain the model
    to produce clean, consistent JSON every time.

    Args:
        catalog_content:  raw catalog text for the product to extract
        price:            product price (appended if not in catalog_content)
        include_few_shot: set False for faster generation when model is
                          already fine-tuned and doesn't need examples

    Returns:
        List of message dicts ready for client.chat.completions.create()
    """
    messages = [{"role": "system", "content": build_system_prompt()}]

    if include_few_shot:
        messages.extend(build_few_shot_messages())

    messages.append({
        "role": "user",
        "content": build_user_prompt(catalog_content, price)
    })

    return messages


def build_retry_messages(
    original_messages: list[dict],
    failed_output: str,
    error_description: str,
) -> list[dict]:
    """
    Build a retry message list when the model's first output failed validation.

    Appends the failed output and the specific error to the conversation,
    then asks the model to fix it. This is the retry mechanism used by
    validate_with_retry() in validator.py.

    Args:
        original_messages:  the original message list from build_full_messages()
        failed_output:      the model's previous (invalid) output
        error_description:  human-readable description of what went wrong

    Returns:
        Extended message list with the error feedback appended.

    Example error_description:
        "JSON parse error: invalid syntax near '}'. Return only raw JSON."
        "Schema error: quantity_unit 'Ounce' is not valid. Use 'oz' instead."
        "Confidence 0.40 is below minimum threshold 0.60. Increase confidence."
    """
    messages = list(original_messages)

    # Append the failed output as if the assistant said it
    messages.append({
        "role": "assistant",
        "content": failed_output
    })

    # Append the error as a user correction
    messages.append({
        "role": "user",
        "content": (
            f"That output failed validation with this error:\n{error_description}\n\n"
            f"Fix the issue and return ONLY the corrected JSON object. "
            f"No explanation, no markdown, no backticks."
        )
    })

    return messages


# ── Token count estimate ──────────────────────────────────────────────────────

def estimate_tokens(messages: list[dict]) -> int:
    """
    Rough token estimate for a list of messages.

    Used by generate_pairs.py to monitor API usage and avoid
    hitting rate limits. Rule of thumb: 1 token ≈ 4 characters.
    Actual token count depends on tokenizer — this is approximate.

    Args:
        messages: list of message dicts

    Returns:
        Estimated token count (integer).
    """
    total_chars = sum(len(m["content"]) for m in messages)
    return total_chars // 4