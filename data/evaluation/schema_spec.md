# ProductEntity Schema Specification
Generated from notebooks/03_schema_design.ipynb

## Dataset
- 75,000 products from Amazon US grocery catalog
- 4 columns: sample_id, catalog_content, image_link, price
- Zero null values

## Required Fields
| Field | Type | Constraint |
|---|---|---|
| product_id | str | always present |
| item_name | str | min 2, max 500 chars |
| price | float | >= 0 |
| quality_score | int | 0-100, computed |
| content_tier | enum | rich/medium/bare, computed |
| extraction_confidence | float | 0.0-1.0, from model |

## Optional Fields
| Field | Type | Coverage | Source |
|---|---|---|---|
| brand | str | ~65% | extracted from item_name |
| category | str | 88.8% | labeling functions |
| quantity_value | float | ~95% | Value field |
| quantity_unit | str | ~94% | Unit field normalized |
| pack_size | int | 26.3% | Pack of N pattern |
| packaging_type | enum | varies | image extraction |
| unit_price | float | 26.3% | derived: price/pack_size |
| dietary_tags | list[str] | ~60% | text extraction |
| allergen_list | list[str] | 38% | text extraction |
| is_organic | bool | varies | derived from dietary_tags |
| is_kosher | bool | varies | derived from dietary_tags |
| description | str | 43.4% | Product Description field |
| bullet_points | list[str] | 72.7% | Bullet Point fields |
| ingredients | list[str] | ~30% | extracted from description |
| packaging_color | str | varies | image extraction |
| image_url | str | 99.99% | image_link column |

## Controlled Vocabularies

### Categories (19)
Coffee & Tea, Breakfast & Cereal, Meat & Seafood, Soups & Canned Goods,
Pasta & Noodles, Bread & Bakery, Protein Bars & Snacks, Supplements & Health,
Grains Beans & Legumes, Oils & Vinegars, Nuts & Seeds, Personal Care & Beauty,
Spices & Seasonings, Condiments & Sauces, Baking & Cooking, Snacks & Candy,
Beverages, Non-Food, Unknown

### Dietary Tags (15)
gluten-free, dairy-free, sugar-free, nut-free, soy-free, vegan, kosher,
organic, non-GMO, keto, paleo, high-protein, low-calorie, caffeine-free, allergen-free

### Units (8)
oz, fl oz, lb, ct, g, kg, ml, L

### Allergens (10)
milk, egg, wheat, soy, peanut, fish, shellfish, sesame, tree_nut, gluten

### Packaging Types (9)
bottle, bag, box, jar, can, sachet, pouch, tube, carton, unknown

## Content Tiers
- Rich: 5+ bullet points AND has description (~43%)
- Medium: has bullet points, no description (~30%)
- Bare: item name only, no bullets (~27%)

## Quality Score Formula (0-100)
- item_name present: +15
- brand extracted: +10
- category assigned: +10
- price present: +5
- quantity_value present: +5
- quantity_unit present: +5
- pack_size present: +5
- packaging_type not unknown: +5
- 5+ bullet points: +15
- 1-4 bullet points: +8
- description present: +10
- dietary_tags present: +8
- allergen_list present: +7

## Validation Rules
1. category must be from 19-category controlled list
2. quantity_unit must be one of 8 canonical units
3. dietary_tags must be from 15-tag controlled list
4. price must be >= 0
5. quality_score must be 0-100
6. extraction_confidence must be 0.0-1.0
7. item_name must be 2-500 characters

## Derived Fields (auto-computed, never extracted)
- unit_price = price / pack_size (when pack_size > 1)
- is_organic = True if "organic" in dietary_tags
- is_kosher = True if "kosher" in dietary_tags

## Files Generated
- data/raw/missing_images.json — 3 missing image IDs
- data/raw/extreme_aspect_ids.json — 25 extreme aspect ratio IDs
- data/raw/human_review_ids.json — 472 dual poor quality IDs
