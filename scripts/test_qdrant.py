"""
scripts/test_qdrant.py
Tests Qdrant search with the newer query_points API.
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from sentence_transformers import SentenceTransformer

load_dotenv()

COLLECTION_NAME = "retailgraph-products"
MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
model  = SentenceTransformer(MODEL_NAME)

# Create payload indexes for filtered search
print("Creating payload indexes...")
from qdrant_client.models import PayloadSchemaType
client.create_payload_index(COLLECTION_NAME, "category",      PayloadSchemaType.KEYWORD)
client.create_payload_index(COLLECTION_NAME, "price",         PayloadSchemaType.FLOAT)
client.create_payload_index(COLLECTION_NAME, "dietary_tags",  PayloadSchemaType.KEYWORD)
client.create_payload_index(COLLECTION_NAME, "allergen_list", PayloadSchemaType.KEYWORD)
print("  ✅ Indexes created")

def search(query, top_k=5, category=None, max_price=None, dietary_tags=None):
    vector = model.encode(query).tolist()

    must_conditions = []
    if category:
        must_conditions.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if max_price is not None:
        must_conditions.append(FieldCondition(key="price", range=Range(lte=max_price)))
    if dietary_tags:
        for tag in dietary_tags:
            must_conditions.append(FieldCondition(key="dietary_tags", match=MatchValue(value=tag)))

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


tests = [
    {"query": "organic green tea",          "kwargs": {}},
    {"query": "spicy sauce for cooking",    "kwargs": {"category": "Condiments & Sauces", "max_price": 15.0}},
    {"query": "protein snack bar",          "kwargs": {"dietary_tags": ["gluten-free"]}},
]

print("── Search Tests ──────────────────────────────────────────")
for t in tests:
    print(f"\nQuery: '{t['query']}' | Filters: {t['kwargs']}")
    for r in search(t["query"], top_k=3, **t["kwargs"]):
        tags = ", ".join(r["dietary_tags"]) if r["dietary_tags"] else "none"
        print(f"  [{r['score']}] {r['item_name']} | {r['brand']} | ${r['price']} | tags: {tags}")
print("──────────────────────────────────────────────────────────")