from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Load domain adapter once at module level ─────────────────────────────────
def _load_adapter(domain: str = "retail") -> dict:
    path = Path(__file__).parent.parent / "domain_adapter" / f"{domain}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_ADAPTER = _load_adapter("retail")

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


# ── Enums ─────────────────────────────────────────────────────────────────────
class ContentTier(str, Enum):
    RICH = "rich"      # 5+ bullets + description
    MEDIUM = "medium"  # has bullets, no description
    BARE = "bare"      # item name only


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


class ImageQuality(str, Enum):
    HIGH = "high"       # shortest side >= 500px
    MEDIUM = "medium"   # shortest side 200-499px
    LOW = "low"         # shortest side < 200px


# ── VisualAttributes — image-only fields ─────────────────────────────────────
class VisualAttributes(BaseModel):
    packaging_type: PackagingType = Field(
        default=PackagingType.UNKNOWN,
        description="Physical packaging format detected from image"
    )
    packaging_color: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Primary packaging color detected from image"
    )
    has_brand_logo: Optional[bool] = Field(
        default=None,
        description="Whether brand logo is visible in image"
    )
    image_quality: ImageQuality = Field(
        default=ImageQuality.HIGH,
        description="Image resolution tier — high/medium/low"
    )

    @field_validator("packaging_color")
    @classmethod
    def clean_color(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        # strip whitespace, lowercase
        v = v.strip().lower()
        if not v:
            return None
        return v


# ── ProductEntity — full product schema ──────────────────────────────────────
class ProductEntity(BaseModel):

    # ── Core Identity ─────────────────────────────────────────────────────────
    product_id: str = Field(
        description="Unique product identifier from sample_id"
    )
    item_name: str = Field(
        min_length=2,
        max_length=500,
        description="Full product name from Item Name field"
    )
    brand: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Brand name extracted from item_name"
    )
    category: Optional[str] = Field(
        default=None,
        description="Product category from controlled vocabulary"
    )

    # ── Physical Properties ───────────────────────────────────────────────────
    quantity_value: Optional[float] = Field(
        default=None,
        ge=0,
        description="Numeric quantity value from Value field"
    )
    quantity_unit: Optional[str] = Field(
        default=None,
        description="Normalized unit — oz, fl oz, lb, ct, g, kg, ml, L"
    )
    pack_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description="Number of units in pack from Pack of N pattern"
    )

    # ── Pricing ───────────────────────────────────────────────────────────────
    price: float = Field(
        ge=0,
        description="Product price in USD"
    )
    unit_price: Optional[float] = Field(
        default=None,
        ge=0,
        description="Price per single unit — derived as price / pack_size"
    )

    # ── Dietary & Compliance ──────────────────────────────────────────────────
    dietary_tags: Optional[list[str]] = Field(
        default=None,
        description="Canonical dietary tags from controlled vocabulary"
    )
    allergen_list: Optional[list[str]] = Field(
        default=None,
        description="Allergens present from controlled vocabulary"
    )
    is_organic: Optional[bool] = Field(
        default=None,
        description="Derived — True if organic in dietary_tags"
    )
    is_kosher: Optional[bool] = Field(
        default=None,
        description="Derived — True if kosher in dietary_tags"
    )

    # ── Content Fields ────────────────────────────────────────────────────────
    description: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Product description text"
    )
    bullet_points: Optional[list[str]] = Field(
        default=None,
        description="List of bullet point strings"
    )
    ingredients: Optional[list[str]] = Field(
        default=None,
        description="Ingredient list extracted from description or bullets"
    )

    # ── Visual Attributes ─────────────────────────────────────────────────────
    visual: Optional[VisualAttributes] = Field(
        default=None,
        description="Image-extracted visual attributes"
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Product image URL"
    )

    # ── Quality & Metadata ────────────────────────────────────────────────────
    quality_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Extraction quality score 0-100"
    )
    content_tier: ContentTier = Field(
        default=ContentTier.BARE,
        description="rich / medium / bare based on content richness"
    )
    extraction_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model confidence score for this extraction"
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("product_id")
    @classmethod
    def product_id_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("product_id cannot be empty")
        return v

    @field_validator("item_name")
    @classmethod
    def clean_item_name(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("item_name too short after stripping whitespace")
        return v

    @field_validator("brand")
    @classmethod
    def clean_brand(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        return v

    @field_validator("category")
    @classmethod
    def category_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if v not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{v}'. "
                f"Must be one of: {VALID_CATEGORIES}"
            )
        return v

    @field_validator("quantity_unit")
    @classmethod
    def unit_must_be_canonical(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if v not in VALID_UNITS:
            raise ValueError(
                f"Non-canonical unit '{v}'. "
                f"Must be one of: {VALID_UNITS}"
            )
        return v

    @field_validator("dietary_tags")
    @classmethod
    def dietary_tags_must_be_valid(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return None
        # remove duplicates, preserve order
        seen = set()
        clean = []
        for tag in v:
            tag = tag.strip()
            if tag not in seen:
                seen.add(tag)
                clean.append(tag)
        # validate against controlled vocabulary
        invalid = [t for t in clean if t not in VALID_DIETARY_TAGS]
        if invalid:
            raise ValueError(
                f"Invalid dietary tags: {invalid}. "
                f"Must be from: {VALID_DIETARY_TAGS}"
            )
        return clean if clean else None

    @field_validator("allergen_list")
    @classmethod
    def allergens_must_be_valid(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return None
        seen = set()
        clean = []
        for allergen in v:
            allergen = allergen.strip().lower()
            if allergen not in seen:
                seen.add(allergen)
                clean.append(allergen)
        invalid = [a for a in clean if a not in VALID_ALLERGENS]
        if invalid:
            raise ValueError(
                f"Invalid allergens: {invalid}. "
                f"Must be from: {VALID_ALLERGENS}"
            )
        return clean if clean else None

    @field_validator("bullet_points")
    @classmethod
    def clean_bullet_points(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return None
        clean = [b.strip() for b in v if b.strip()]
        return clean if clean else None

    @field_validator("ingredients")
    @classmethod
    def clean_ingredients(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return None
        clean = [i.strip() for i in v if i.strip()]
        return clean if clean else None

    # ── Derived fields — computed after all fields set ────────────────────────
    @model_validator(mode="after")
    def compute_derived_fields(self) -> "ProductEntity":
        # unit_price
        if self.price and self.pack_size and self.pack_size > 1:
            self.unit_price = round(self.price / self.pack_size, 4)

        # is_organic and is_kosher from dietary_tags
        if self.dietary_tags:
            self.is_organic = "organic" in self.dietary_tags
            self.is_kosher = "kosher" in self.dietary_tags

        return self

    # ── Content tier assignment ───────────────────────────────────────────────
    def assign_content_tier(self) -> None:
        bullet_count = len(self.bullet_points) if self.bullet_points else 0
        has_description = self.description is not None
        if bullet_count >= 5 and has_description:
            self.content_tier = ContentTier.RICH
        elif bullet_count >= 1:
            self.content_tier = ContentTier.MEDIUM
        else:
            self.content_tier = ContentTier.BARE

    # ── Quality score computation ─────────────────────────────────────────────
    def compute_quality_score(self) -> None:
        score = 0
        if self.item_name:                                          score += 15
        if self.brand:                                              score += 10
        if self.category:                                           score += 10
        if self.price is not None:                                  score += 5
        if self.quantity_value:                                     score += 5
        if self.quantity_unit:                                      score += 5
        if self.pack_size:                                          score += 5
        if self.visual and self.visual.packaging_type != PackagingType.UNKNOWN:
            score += 5
        bullets = self.bullet_points or []
        if len(bullets) >= 5:                                       score += 15
        elif len(bullets) >= 1:                                     score += 8
        if self.description:                                        score += 10
        if self.dietary_tags:                                       score += 8
        if self.allergen_list:                                      score += 7
        self.quality_score = min(score, 100)