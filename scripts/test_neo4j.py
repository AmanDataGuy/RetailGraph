from dotenv import load_dotenv
import os
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(username, password))

try:
    driver.verify_connectivity()
    print("✅ Neo4j connection successful!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
finally:
    driver.close()