import json
from collections import Counter

ALLOWED_CATEGORIES = [
    "Beverages", "Coffee & Tea", "Snacks & Candy", "Condiments & Sauces",
    "Grains, Beans & Legumes", "Baking & Cooking", "Spices & Seasonings",
    "Supplements & Health", "Nuts & Seeds", "Personal Care & Beauty",
    "Protein Bars & Snacks", "Dairy & Eggs", "Meat & Seafood",
    "Fruits & Vegetables", "Frozen Foods", "Bakery & Bread",
    "Baby & Kids", "Household & Cleaning", "Personal Care",
    "Pet Supplies", "Unknown"
]

ALLOWED_UNITS = ["oz", "fl oz", "lb", "g", "kg", "ml", "l", "ct", "pack"]

REQUIRED_FIELDS = ["item_name", "price", "quantity_value", "quantity_unit",
                   "category", "dietary_tags", "allergen_list", "extraction_confidence"]

print("=" * 60)
print("Validating train_r3.jsonl...")
print("=" * 60)

total = 0
parse_errors = 0
missing_fields = 0
empty_assistant = 0
invalid_json_assistant = 0
invalid_category = 0
invalid_unit = 0
bad_messages = 0

category_counter = Counter()
unit_counter = Counter()

with open('data/training/train_r3.jsonl', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        
        try:
            ex = json.loads(line)
        except json.JSONDecodeError as e:
            parse_errors += 1
            print(f"  Line {i+1}: JSON parse error: {e}")
            continue
        
        total += 1
        
        # Check structure
        if 'messages' not in ex:
            bad_messages += 1
            continue
        
        messages = ex['messages']
        roles = [m['role'] for m in messages]
        
        # Must have system, user, assistant at minimum
        if 'system' not in roles or 'user' not in roles or 'assistant' not in roles:
            bad_messages += 1
            continue
        
        # Get last assistant message
        assistant_msgs = [m for m in messages if m['role'] == 'assistant']
        last_assistant = assistant_msgs[-1]
        content = last_assistant.get('content', '')
        
        if isinstance(content, list):
            text = ' '.join(c.get('text', '') for c in content if c.get('type') == 'text')
        else:
            text = content
        
        if not text.strip():
            empty_assistant += 1
            continue
        
        # Try parsing assistant JSON
        try:
            # Strip markdown if present
            clean = text.strip()
            if '```' in clean:
                clean = clean.split('```')[1]
                if clean.startswith('json'):
                    clean = clean[4:]
            pred = json.loads(clean.strip())
        except json.JSONDecodeError:
            invalid_json_assistant += 1
            if invalid_json_assistant <= 3:
                print(f"  Line {i+1}: Invalid assistant JSON: {text[:100]}")
            continue
        
        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in pred:
                missing_fields += 1
                break
        
        # Check category
        cat = pred.get('category', '')
        if cat and cat not in ALLOWED_CATEGORIES:
            invalid_category += 1
            if invalid_category <= 5:
                print(f"  Line {i+1}: Invalid category: {cat}")
        category_counter[cat] += 1
        
        # Check unit
        unit = pred.get('quantity_unit', '')
        if unit and unit not in ALLOWED_UNITS:
            invalid_unit += 1
        unit_counter[unit] += 1

print(f"\nTotal examples:          {total}")
print(f"Parse errors:            {parse_errors}")
print(f"Bad message structure:   {bad_messages}")
print(f"Empty assistant:         {empty_assistant}")
print(f"Invalid assistant JSON:  {invalid_json_assistant}")
print(f"Missing required fields: {missing_fields}")
print(f"Invalid categories:      {invalid_category}")
print(f"Invalid units:           {invalid_unit}")

print(f"\n{'='*60}")
print("Category distribution:")
for cat, count in sorted(category_counter.items(), key=lambda x: -x[1])[:15]:
    print(f"  {cat:<35} {count:>5}")

print(f"\nUnit distribution:")
for unit, count in sorted(unit_counter.items(), key=lambda x: -x[1]):
    print(f"  {unit:<15} {count:>5}")

print(f"\n{'='*60}")
if parse_errors == 0 and bad_messages == 0 and empty_assistant == 0 and invalid_json_assistant == 0:
    print("✅ File looks CLEAN — safe to train on")
else:
    print("⚠️  Issues found — review before training")