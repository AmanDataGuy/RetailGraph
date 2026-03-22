"""
verify_extractions.py

Verifies and corrects 'category' and 'dietary_tags' fields in the 2,161
high-confidence extractions using GPT-4o-mini.

Cost: ~$0.15 total
Time: ~10-15 minutes

Usage:
    pip install openai
    set OPENAI_API_KEY=sk-...
    python verify_extractions.py

Output:
    data/training/verified_extractions.jsonl  (2,161 corrected records)
"""

import json
import os
import time
from pathlib import Path
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────
INPUT_FILE  = "extraction"
OUTPUT_FILE = "data/training/verified_extractions.jsonl"
BATCH_SIZE  = 10   # process 10 at a time to reduce API calls
SLEEP_SEC   = 0.5  # sleep between batches to avoid rate limits

ALLOWED_CATEGORIES = [
    "Beverages",
    "Snacks & Candy",
    "Dairy & Eggs",
    "Meat & Seafood",
    "Fruits & Vegetables",
    "Grains, Beans & Legumes",
    "Condiments & Sauces",
    "Spices & Seasonings",
    "Frozen Foods",
    "Bakery & Bread",
    "Coffee & Tea",
    "Supplements & Health",
    "Baby & Kids",
    "Household & Cleaning",
    "Personal Care",
    "Pet Supplies",
    "Unknown",
]

ALLOWED_DIETARY_TAGS = [
    "organic", "vegan", "vegetarian", "gluten-free", "dairy-free",
    "nut-free", "soy-free", "egg-free", "kosher", "halal",
    "non-GMO", "sugar-free", "low-sodium", "caffeine-free",
    "high-protein", "keto-friendly",
]

SYSTEM_PROMPT = f"""You are a grocery product data verifier.

Given a product listing and its current category and dietary_tags, 
verify if they are correct and fix any errors.

ALLOWED CATEGORIES (use EXACTLY one of these):
{json.dumps(ALLOWED_CATEGORIES, indent=2)}

ALLOWED DIETARY TAGS (use ONLY values from this list, empty list if none apply):
{json.dumps(ALLOWED_DIETARY_TAGS, indent=2)}

Return ONLY a JSON array where each element has:
  {{"id": "sample_id", "category": "corrected category", "dietary_tags": ["tag1", "tag2"]}}

No explanation, no markdown, no backticks. Pure JSON array only."""


def build_batch_prompt(batch):
    """Build a prompt for a batch of products."""
    products = []
    for ex in batch:
        pred = ex["prediction"]
        products.append({
            "id": ex["sample_id"],
            "product": ex["catalog_content"][:300],  # truncate to save tokens
            "current_category": pred.get("category", "Unknown"),
            "current_dietary_tags": pred.get("dietary_tags", []),
        })
    return f"Verify and correct these {len(batch)} products:\n{json.dumps(products, indent=2)}"


def verify_batch(client, batch):
    """Send a batch to GPT-4o-mini and get corrected fields."""
    prompt = build_batch_prompt(batch)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
        max_tokens=1000,
    )
    
    text = response.choices[0].message.content.strip()
    
    # Strip markdown if present
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    
    return json.loads(text.strip())


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable first.")
    
    client = OpenAI(api_key=api_key)
    
    # Load high_conf extractions
    print("Loading high_conf extractions...")
    high_conf = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                if ex["bucket"] == "high_conf" and ex["prediction"]:
                    high_conf.append(ex)
            except Exception:
                continue
    
    print(f"Loaded {len(high_conf)} high_conf extractions.")
    
    # Process in batches
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    corrected = 0
    errors = 0
    
    # Build correction map
    corrections = {}  # sample_id -> {category, dietary_tags}
    
    total_batches = (len(high_conf) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(high_conf), BATCH_SIZE):
        batch = high_conf[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        try:
            results = verify_batch(client, batch)
            for r in results:
                corrections[str(r["id"])] = {
                    "category": r["category"],
                    "dietary_tags": r["dietary_tags"],
                }
                corrected += 1
        except Exception as e:
            print(f"  Batch {batch_num} error: {e} — keeping original values")
            for ex in batch:
                corrections[ex["sample_id"]] = {
                    "category": ex["prediction"].get("category", "Unknown"),
                    "dietary_tags": ex["prediction"].get("dietary_tags", []),
                }
            errors += 1
        
        if batch_num % 10 == 0 or batch_num == total_batches:
            print(f"  [{batch_num}/{total_batches}] {corrected} corrected, {errors} errors")
        
        time.sleep(SLEEP_SEC)
    
    # Write corrected extractions
    print(f"\nWriting corrected extractions to {OUTPUT_FILE}...")
    written = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for ex in high_conf:
            pred = ex["prediction"].copy()
            sid = ex["sample_id"]
            
            if sid in corrections:
                # Validate category
                cat = corrections[sid]["category"]
                if cat not in ALLOWED_CATEGORIES:
                    cat = "Unknown"
                pred["category"] = cat
                
                # Validate dietary_tags
                tags = [t for t in corrections[sid]["dietary_tags"] if t in ALLOWED_DIETARY_TAGS]
                pred["dietary_tags"] = tags
            
            ex["prediction"] = pred
            f_out.write(json.dumps(ex) + "\n")
            written += 1
    
    print(f"Done! Written {written} verified extractions to {OUTPUT_FILE}")
    print(f"Corrected: {corrected} | Errors (kept original): {errors}")
    print(f"\nNext step: run prepare_round2.py to merge with train.jsonl")


if __name__ == "__main__":
    main()