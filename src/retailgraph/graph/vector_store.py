"""
# vector_store.py - embeds products into the Qdrant collection used for semantic search

- Holds the embed/upsert pipeline and a filtered `search()` (used by scripts/run_benchmark.py).
- Run: `python -m retailgraph.graph.vector_store` (full upsert) or `... --test` (searches only).
"""

import argparse
import hashlib
import json
import os

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

load_dotenv()

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "retailgraph-products"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE     = 384
BATCH_SIZE      = 100


# Qdrant client; timeout=60 because the ~5s default times out batched upserts to a cloud cluster.
def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)


# Create the collection and its payload indexes if it doesn't exist yet.
def create_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"  Collection '{COLLECTION_NAME}' already exists — skipping create.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"  ✅ Collection '{COLLECTION_NAME}' created.")
    client.create_payload_index(COLLECTION_NAME, "category",      PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION_NAME, "price",         PayloadSchemaType.FLOAT)
    client.create_payload_index(COLLECTION_NAME, "dietary_tags",  PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION_NAME, "allergen_list", PayloadSchemaType.KEYWORD)
    print("  ✅ Payload indexes created.")


# Load high-confidence extractions from the JSONL file.
def load_products(input_file: str) -> list[dict]:
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


# Embed "item_name. catalog_content" (first 512 chars) and build Qdrant points with payloads.
def build_points(products: list[dict], model: TextEmbedding) -> list[PointStruct]:
    texts = []
    for p in products:
        item_name = p.get("prediction", {}).get("item_name") or ""
        texts.append(f"{item_name}. {p.get('catalog_content') or ''}".strip()[:512])

    print(f"  Embedding {len(texts)} products...")
    vectors = list(model.embed(texts, batch_size=32))

    points = []
    for i, (product, vector) in enumerate(zip(products, vectors, strict=False)):
        pred = product.get("prediction", {})
        sid  = product.get("sample_id", str(i))

        points.append(PointStruct(
            # sha256, not hash(): hash() is randomized per process and would orphan old points.
            id=int(hashlib.sha256(sid.encode()).hexdigest()[:15], 16),
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


# Upsert all points in batches and print the collection summary.
def upsert_all(client: QdrantClient, points: list[PointStruct]) -> None:
    total = len(points)
    print(f"\nUpserting {total} vectors to Qdrant (batch={BATCH_SIZE})...")

    for i in range(0, total, BATCH_SIZE):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i:i + BATCH_SIZE])
        print(f"  [{min(i + BATCH_SIZE, total)}/{total}] upserted")

    info = client.get_collection(COLLECTION_NAME)
    print("\n✅ Qdrant collection ready.")
    print(f"  Vectors stored: {info.points_count}")
    print(f"  Vector size:    {VECTOR_SIZE}")
    print("  Distance:       Cosine")


# Pure Qdrant search: must have ALL dietary_tags, NONE of exclude_tags, plus category/price filters.
def search(
    query: str,
    client: QdrantClient,
    model: TextEmbedding,
    top_k: int = 5,
    category: str = None,
    max_price: float = None,
    min_price: float = None,
    dietary_tags: list[str] = None,
    exclude_tags: list[str] = None,
) -> list[dict]:
    vector = list(model.embed([query]))[0].tolist()

    must_conditions = []
    if category:
        must_conditions.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if max_price is not None or min_price is not None:
        must_conditions.append(
            FieldCondition(key="price", range=Range(lte=max_price, gte=min_price))
        )
    must_conditions += [FieldCondition(key="dietary_tags", match=MatchValue(value=tag))
                        for tag in dietary_tags or []]
    must_not_conditions = [FieldCondition(key="dietary_tags", match=MatchValue(value=tag))
                           for tag in exclude_tags or []]

    query_filter = (
        Filter(must=must_conditions or None, must_not=must_not_conditions or None)
        if (must_conditions or must_not_conditions) else None
    )

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


# Run three sample searches to check the collection works.
def run_test(client: QdrantClient, model: TextEmbedding) -> None:
    tests = [
        {"query": "organic green tea", "kwargs": {}},
        {"query": "spicy sauce for cooking",
         "kwargs": {"category": "Condiments & Sauces", "max_price": 15.0}},
        {"query": "protein snack bar", "kwargs": {"dietary_tags": ["gluten-free"]}},
    ]

    print("\n── Search Tests ──────────────────────────────────────────")
    for t in tests:
        print(f"\nQuery: '{t['query']}' | Filters: {t['kwargs']}")
        results = search(t["query"], client, model, top_k=3, **t["kwargs"])
        for r in results:
            tags = ", ".join(r["dietary_tags"]) if r["dietary_tags"] else "none"
            print(f"  [{r['score']}] {r['item_name']} | {r['brand']} | ${r['price']} | {r['category']} | tags: {tags}")
    print("──────────────────────────────────────────────────────────")


# CLI entry: full embed + upsert, or just the test searches with --test.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training/verified_extractions.jsonl")
    parser.add_argument("--test", action="store_true", help="Run search tests only")
    args = parser.parse_args()

    print("Loading embedding model...")
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    print(f"  ✅ Model loaded: {EMBEDDING_MODEL}")

    client = get_client()
    print(f"  ✅ Connected to Qdrant: {QDRANT_URL}")

    if args.test:
        run_test(client, model)
        return

    create_collection(client)

    print(f"\nLoading products from {args.input}...")
    products = load_products(args.input)
    print(f"  Loaded: {len(products)} products")

    points = build_points(products, model)
    upsert_all(client, points)
    run_test(client, model)


if __name__ == "__main__":
    main()