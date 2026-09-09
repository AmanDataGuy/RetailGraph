"""
RetailGraph — infra health check for CI (.github/workflows/health-check.yml).

Checks Neo4j, Qdrant, and Groq connectivity directly — not via the deployed
API, so this catches infra failures (e.g. Neo4j Aura auto-pausing, which
happened twice during development) even before/without a live deployment.
Exits non-zero on any failure, which the workflow uses to trigger an email.

Usage:
    python scripts/health_check.py
"""
import os
import sys

from dotenv import load_dotenv
load_dotenv()

failures = []

print("=== Neo4j ===")
try:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        session.run("RETURN 1").single()
    driver.close()
    print("  OK")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    failures.append("Neo4j")

print("=== Qdrant ===")
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    client.get_collections()
    print("  OK")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    failures.append("Qdrant")

print("=== Groq ===")
try:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
    )
    print("  OK")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    failures.append("Groq")

if failures:
    print(f"\nFAILED: {', '.join(failures)}")
    sys.exit(1)

print("\nAll services healthy.")
