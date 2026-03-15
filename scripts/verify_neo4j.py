"""
scripts/verify_neo4j.py
Run after adding Neo4j credentials to .env
"""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI      = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

print(f"Connecting to: {URI}")
print(f"Username:      {USERNAME}")
print(f"Database:      {DATABASE}")
print()

try:
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    driver.verify_connectivity()
    print("✅ Connected to Neo4j AuraDB successfully!")

    with driver.session(database=DATABASE) as session:
        result = session.run("RETURN 'RetailGraph is ready!' AS message")
        print(f"✅ Query test: {result.single()['message']}")

    driver.close()

except Exception as e:
    print(f"❌ Connection failed: {e}")