"""
# schemas.py - Pydantic schema for one extracted product (ProductEntity)

- Holds the controlled vocabularies, enums, field validators, derived fields and the quality score.
- Used by validator.py to check teacher/model JSON, and by graph/builder.py and tests.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

_ADAPTER_PATH = Path(__file__).parent.parent / "domain_adapter" / "retail.yaml"
with open(_ADAPTER_PATH, encoding="utf-8") as _f:
    _ADAPTER = yaml.safe_load(_f)

VALID_CATEGORIES: list[str] = [
    "Coffee & Tea", "Breakfast & Cereal", "Meat & Seafood",
    "Soups & Canned Goods", "Pasta & Noodles", "Bread & Bakery",
    "Protein Bars & Snacks", "Supplements & Health",
    "Grains, Beans & Legumes", "Oils & Vinegars", "Nuts & Seeds",
    "Personal Care & Beauty", "Spices & Seasonings",
    "Condiments & Sauces", "Baking & Cooking", "Snacks & Candy",
    "Beverages", "Non-Food", "Unknown",
]
VALID_UNITS: list[str] = ["oz", "fl oz", "lb", "ct", "g", "kg", "ml", "L"]
VALID_DIETARY_TAGS: list[str] = _ADAPTER.get("dietary_tags_controlled", [])
VALID_ALLERGENS: list[str] = _ADAPTER.get("allergens_controlled", [])


# How much text a product listing has.
class ContentTier(str, Enum):
    RICH = "rich"      # 5+ bullets + description
    MEDIUM = "medium"  # has bullets, no description
    BARE = "bare"      # item name only


# Physical packaging seen in the product image.
class PackagingType(str, Enum):
    BOTTLE = "bottle"
    BAG = "bag"
    BOX = "box"
    JAR = "jar"
    CAN = "can"
    SACHET = "sachet"
    POUCH = "pouch"
    TUBE = "tube"
    CARTON = "carton"
    UNKNOWN = "unknown"


# Image resolution tier by shortest side: high >= 500px, medium 200-499px, low < 200px.
class ImageQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Fields that can only come from the product image.
class VisualAttributes(BaseModel):
    packaging_type: PackagingType = Field(
        default=PackagingType.UNKNOWN, description="Physical packaging format detected from image")
    packaging_color: str | None = Field(
        default=None, max_length=50, description="Primary packaging color detected from image")
    has_brand_logo: bool | None = Field(default=None, description="Whether brand logo is visible in image")
    image_quality: ImageQuality = Field(
        default=ImageQuality.HIGH, description="Image resolution tier — high/medium/low")

    # Lowercase and trim the colour; blank becomes None.
    @field_validator("packaging_color")
    @classmethod
    def clean_color(cls, v: str | None) -> str | None:
        return None if v is None else (v.strip().lower() or None)


# Strip each string and drop duplicates, keeping the first-seen order.
def _dedupe(items: list[str], lower: bool = False) -> list[str]:
    return list(dict.fromkeys(i.strip().lower() if lower else i.strip() for i in items))


# Strip each string and drop blanks; an empty result becomes None.
def _strip_nonblank(items: list[str] | None) -> list[str] | None:
    if items is None:
        return None
    return [i.strip() for i in items if i.strip()] or None


# One product with all extracted fields; validators enforce the controlled vocabularies.
class ProductEntity(BaseModel):
    # Core identity
    product_id: str = Field(description="Unique product identifier from sample_id")
    item_name: str = Field(min_length=2, max_length=500, description="Full product name from Item Name field")
    brand: str | None = Field(default=None, max_length=100, description="Brand name extracted from item_name")
    category: str | None = Field(default=None, description="Product category from controlled vocabulary")

    # Physical properties
    quantity_value: float | None = Field(default=None, ge=0, description="Numeric quantity value from Value field")
    quantity_unit: str | None = Field(
        default=None, description="Normalized unit — oz, fl oz, lb, ct, g, kg, ml, L")
    pack_size: int | None = Field(
        default=None, ge=1, le=1000, description="Number of units in pack from Pack of N pattern")

    # Pricing
    price: float = Field(ge=0, description="Product price in USD")
    unit_price: float | None = Field(
        default=None, ge=0, description="Price per single unit — derived as price / pack_size")

    # Dietary & compliance
    dietary_tags: list[str] = Field(
        default_factory=list, description="Canonical dietary tags from controlled vocabulary")
    allergen_list: list[str] = Field(
        default_factory=list, description="Allergens present from controlled vocabulary")
    is_organic: bool | None = Field(default=None, description="Derived — True if organic in dietary_tags")
    is_kosher: bool | None = Field(default=None, description="Derived — True if kosher in dietary_tags")

    # Content
    description: str | None = Field(default=None, max_length=5000, description="Product description text")
    bullet_points: list[str] | None = Field(default=None, description="List of bullet point strings")
    ingredients: list[str] | None = Field(
        default=None, description="Ingredient list extracted from description or bullets")

    # Visual
    visual: VisualAttributes | None = Field(default=None, description="Image-extracted visual attributes")
    image_url: str | None = Field(default=None, description="Product image URL")

    # Quality & metadata
    quality_score: int = Field(default=0, ge=0, le=100, description="Extraction quality score 0-100")
    content_tier: ContentTier = Field(
        default=ContentTier.BARE, description="rich / medium / bare based on content richness")
    extraction_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Model confidence score for this extraction")

    # Treat a null tag/allergen list as empty before type checking.
    @field_validator("dietary_tags", "allergen_list", mode="before")
    @classmethod
    def coerce_none_to_list(cls, v):
        return [] if v is None else v

    # Reject a blank product_id.
    @field_validator("product_id")
    @classmethod
    def product_id_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("product_id cannot be empty")
        return v

    # Reject an item_name shorter than 2 characters once trimmed.
    @field_validator("item_name")
    @classmethod
    def clean_item_name(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("item_name too short after stripping whitespace")
        return v

    # Trim the brand; blank becomes None.
    @field_validator("brand")
    @classmethod
    def clean_brand(cls, v: str | None) -> str | None:
        return None if v is None else (v.strip() or None)

    # Category must be in VALID_CATEGORIES.
    @field_validator("category")
    @classmethod
    def category_must_be_valid(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category '{v}'. Must be one of: {VALID_CATEGORIES}")
        return v

    # Unit must already be one of the canonical VALID_UNITS.
    @field_validator("quantity_unit")
    @classmethod
    def unit_must_be_canonical(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_UNITS:
            raise ValueError(f"Non-canonical unit '{v}'. Must be one of: {VALID_UNITS}")
        return v

    # Dedupe dietary tags and check each against the controlled list; empty becomes None.
    @field_validator("dietary_tags")
    @classmethod
    def dietary_tags_must_be_valid(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        clean = _dedupe(v)
        invalid = [t for t in clean if t not in VALID_DIETARY_TAGS]
        if invalid:
            raise ValueError(f"Invalid dietary tags: {invalid}. Must be from: {VALID_DIETARY_TAGS}")
        return clean or None

    # Lowercase and dedupe allergens and check each against the controlled list; empty becomes None.
    @field_validator("allergen_list")
    @classmethod
    def allergens_must_be_valid(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        clean = _dedupe(v, lower=True)
        invalid = [a for a in clean if a not in VALID_ALLERGENS]
        if invalid:
            raise ValueError(f"Invalid allergens: {invalid}. Must be from: {VALID_ALLERGENS}")
        return clean or None

    # Drop blank bullet points.
    @field_validator("bullet_points")
    @classmethod
    def clean_bullet_points(cls, v: list[str] | None) -> list[str] | None:
        return _strip_nonblank(v)

    # Drop blank ingredients.
    @field_validator("ingredients")
    @classmethod
    def clean_ingredients(cls, v: list[str] | None) -> list[str] | None:
        return _strip_nonblank(v)

    # Fill unit_price, is_organic and is_kosher from the other fields.
    @model_validator(mode="after")
    def compute_derived_fields(self) -> ProductEntity:
        if self.price and self.pack_size and self.pack_size > 1:
            self.unit_price = round(self.price / self.pack_size, 4)
        if self.dietary_tags:
            self.is_organic = "organic" in self.dietary_tags
            self.is_kosher = "kosher" in self.dietary_tags
        return self

    # Set content_tier from how many bullets and whether there is a description.
    def assign_content_tier(self) -> None:
        bullet_count = len(self.bullet_points) if self.bullet_points else 0
        if bullet_count >= 5 and self.description is not None:
            self.content_tier = ContentTier.RICH
        elif bullet_count >= 1:
            self.content_tier = ContentTier.MEDIUM
        else:
            self.content_tier = ContentTier.BARE

    # Score 0-100 by adding fixed points for each field that is filled in.
    def compute_quality_score(self) -> None:
        bullets = self.bullet_points or []
        score = (
            15 * bool(self.item_name)
            + 10 * bool(self.brand)
            + 10 * bool(self.category)
            + 5 * (self.price is not None)
            + 5 * bool(self.quantity_value)
            + 5 * bool(self.quantity_unit)
            + 5 * bool(self.pack_size)
            + 5 * bool(self.visual and self.visual.packaging_type != PackagingType.UNKNOWN)
            + (15 if len(bullets) >= 5 else 8 if bullets else 0)
            + 10 * bool(self.description)
            + 8 * bool(self.dietary_tags)
            + 7 * bool(self.allergen_list)
        )
        self.quality_score = min(score, 100)
