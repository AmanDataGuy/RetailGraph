"""
src/graph/vector_store.py

Embeds product catalog_content using sentence-transformers
and upserts into Qdrant for semantic similarity search.

Usage:
    python src/graph/vector_store.py              # full upsert
    python src/graph/vector_store.py --test       # test search only
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    PayloadSchemaType
)
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "retailgraph-products"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE     = 384
BATCH_SIZE      = 100


# ── Client & Collection ───────────────────────────────────────────────────────

def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def create_collection(client: QdrantClient) -> None:
    """Create collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"  Collection '{COLLECTION_NAME}' already exists — skipping create.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        )
    )
    print(f"  ✅ Collection '{COLLECTION_NAME}' created.")
    # Create payload indexes for filtered search
    client.create_payload_index(COLLECTION_NAME, "category",      PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION_NAME, "price",         PayloadSchemaType.FLOAT)
    client.create_payload_index(COLLECTION_NAME, "dietary_tags",  PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION_NAME, "allergen_list", PayloadSchemaType.KEYWORD)
    print("  ✅ Payload indexes created.")


# ── Load & Embed ──────────────────────────────────────────────────────────────

def load_products(input_file: str) -> list[dict]:
    """Load verified extractions."""
    products = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                if ex.get("prediction") and ex.get("bucket") == "high_conf":
                    products.append(ex)
            except Exception:
                continue
    return products


def build_points(products: list[dict], model: SentenceTransformer) -> list[PointStruct]:
    """Embed catalog_content and build Qdrant points."""
    texts = []
    for p in products:
        catalog = p.get("catalog_content") or ""
        pred    = p.get("prediction", {})
        # Enrich embedding text with item_name for better semantic matching
        item_name = pred.get("item_name") or ""
        text = f"{item_name}. {catalog}".strip()
        texts.append(text[:512])  # truncate to avoid token limits

    print(f"  Embedding {len(texts)} products...")
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True)

    points = []
    for i, (product, vector) in enumerate(zip(products, vectors)):
        pred = product.get("prediction", {})
        sid  = product.get("sample_id", str(i))

        points.append(PointStruct(
            id=abs(hash(sid)) % (2**53),  # Qdrant needs integer or UUID
            vector=vector.tolist(),
            payload={
                "product_id":    pred.get("product_id") or sid,
                "sample_id":     sid,
                "item_name":     pred.get("item_name") or "",
                "brand":         pred.get("brand") or "",
                "category":      pred.get("category") or "Unknown",
                "price":         float(pred.get("price") or 0),
                "quantity_value": pred.get("quantity_value"),
                "quantity_unit": pred.get("quantity_unit") or "",
                "dietary_tags":  pred.get("dietary_tags") or [],
                "allergen_list": pred.get("allergen_list") or [],
                "quality_score": int(pred.get("extraction_confidence", 0) * 100),
            }
        ))

    return points


# ── Upsert ────────────────────────────────────────────────────────────────────

def upsert_all(client: QdrantClient, points: list[PointStruct]) -> None:
    """Upsert all points in batches."""
    total = len(points)
    print(f"\nUpserting {total} vectors to Qdrant (batch={BATCH_SIZE})...")

    for i in range(0, total, BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        end = min(i + BATCH_SIZE, total)
        print(f"  [{end}/{total}] upserted")

    info = client.get_collection(COLLECTION_NAME)
    print(f"\n✅ Qdrant collection ready.")
    print(f"  Vectors stored: {info.points_count}")
    print(f"  Vector size:    {VECTOR_SIZE}")
    print(f"  Distance:       Cosine")


# ── Search ────────────────────────────────────────────────────────────────────

def search(
    query: str,
    client: QdrantClient,
    model: SentenceTransformer,
    top_k: int = 5,
    category: str = None,
    max_price: float = None,
    dietary_tags: list[str] = None,
) -> list[dict]:
    """
    Semantic search with optional filters.

    Args:
        query:        Natural language query
        top_k:        Number of results
        category:     Filter by category (exact match)
        max_price:    Filter by max price
        dietary_tags: Filter — product must have ALL these tags
    """
    vector = model.encode(query).tolist()

    # Build filters
    must_conditions = []

    if category:
        must_conditions.append(
            FieldCondition(key="category", match=MatchValue(value=category))
        )

    if max_price is not None:
        must_conditions.append(
            FieldCondition(key="price", range=Range(lte=max_price))
        )

    if dietary_tags:
        for tag in dietary_tags:
            must_conditions.append(
                FieldCondition(key="dietary_tags", match=MatchValue(value=tag))
            )

    query_filter = Filter(must=must_conditions) if must_conditions else None

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "score":      round(r.score, 3),
            "item_name":  r.payload.get("item_name"),
            "brand":      r.payload.get("brand"),
            "category":   r.payload.get("category"),
            "price":      r.payload.get("price"),
            "dietary_tags": r.payload.get("dietary_tags"),
        }
        for r in results
    ]


# ── Test ──────────────────────────────────────────────────────────────────────

def run_test(client: QdrantClient, model: SentenceTransformer) -> None:
    """Run 3 test searches to verify everything works."""
    tests = [
        {
            "query": "organic green tea",
            "kwargs": {},
        },
        {
            "query": "spicy sauce for cooking",
            "kwargs": {"category": "Condiments & Sauces", "max_price": 15.0},
        },
        {
            "query": "protein snack bar",
            "kwargs": {"dietary_tags": ["gluten-free"]},
        },
    ]

    print("\n── Search Tests ──────────────────────────────────────────")
    for t in tests:
        print(f"\nQuery: '{t['query']}' | Filters: {t['kwargs']}")
        results = search(t["query"], client, model, top_k=3, **t["kwargs"])
        for r in results:
            tags = ", ".join(r["dietary_tags"]) if r["dietary_tags"] else "none"
            print(f"  [{r['score']}] {r['item_name']} | {r['brand']} | ${r['price']} | {r['category']} | tags: {tags}")
    print("──────────────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training/verified_extractions.jsonl")
    parser.add_argument("--test", action="store_true", help="Run search tests only")
    args = parser.parse_args()

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  ✅ Model loaded: {EMBEDDING_MODEL}")

    client = get_client()
    print(f"  ✅ Connected to Qdrant: {QDRANT_URL}")

    if args.test:
        run_test(client, model)
        return

    # Full upsert flow
    create_collection(client)

    print(f"\nLoading products from {args.input}...")
    products = load_products(args.input)
    print(f"  Loaded: {len(products)} products")

    points = build_points(products, model)
    upsert_all(client, points)
    run_test(client, model)


if __name__ == "__main__":
    main()