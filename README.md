<div align="center">

# 🕸️ RetailGraph
### Deterministic Retail Reasoning via Knowledge Graph Construction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-018bff?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![LangChain](https://img.shields.io/badge/🦜🔗-LangChain-green)](https://langchain.com/)
[![OpenAI](https://img.shields.io/badge/GPT-4o--mini-green?logo=openai&logoColor=white)](https://openai.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

**"Vectors guess. Graphs know."**

[View Demo](#) · [Report Bug](issues) · [Request Feature](issues)

</div>

---

## 🚀 Overview

**RetailGraph** is an enterprise-grade GraphRAG (Retrieval Augmented Generation) system designed to eliminate Large Language Model (LLM) hallucinations in e-commerce applications.

Unlike standard vector search engines that rely on probabilistic similarity, RetailGraph builds a **deterministic Knowledge Graph (KG)** from unstructured product catalogs. This allows for complex, multi-hop reasoning queries (e.g., *"Show me gluten-free snacks under $5 that are not spicy"*) with **100% factual accuracy**.

### ⚡ The Problem vs. The Solution

| Feature | Standard Vector RAG ❌ | RetailGraph (GraphRAG) ✅ |
| :--- | :--- | :--- |
| **Reasoning** | Probabilistic (Guessing) | Deterministic (Logic-based) |
| **Accuracy** | Prone to Hallucinations | Exact Schema Alignment |
| **Query Type** | "Find similar items..." | "Find items where Brand=X AND Price<Y" |
| **Data Structure** | Unstructured Chunks | Structured Entities & Relationships |

---

## 🏗️ System Architecture

RetailGraph employs a multi-stage **ETL (Extract, Transform, Load)** pipeline to weave unstructured text into a semantic web of data.

```mermaid
graph TD
    subgraph Ingestion ["Phase 1: Structured Extraction"]
    A[Raw CSV Data] -->|Load| B(Pandas DataFrame)
    B -->|LLM + Pydantic| C[Entity Extraction Agent]
    C -->|Validates| D{Schema Check}
    D -- Fail --> C
    D -- Pass --> E[Structured JSON]
    end

    subgraph Storage ["Phase 2: Knowledge Graph Construction"]
    E -->|Cypher Injection| F[(Neo4j AuraDB)]
    F -->|Nodes| N1(Product)
    F -->|Nodes| N2(Brand)
    F -->|Nodes| N3(Ingredient)
    F -->|Edges| R1[:MANUFACTURED_BY]
    F -->|Edges| R2[:CONTAINS]
    end

    subgraph Inference ["Phase 3: GraphRAG Inference"]
    User[User Query] -->|Natural Language| G[LangChain Agent]
    G -->|Schema Awareness| H[Text-to-Cypher Translator]
    H -->|Execute| F
    F -->|Graph Context| I[LLM Reasoner]
    I -->|Final Answer| User
    end
```

## 🧠 The Ontology

To ground the LLM, we define a strict schema. The database isn't just a blob of text; it is a connected ecosystem of entities.

```mermaid
classDiagram
    Product "1" --> "1" Brand : MANUFACTURED_BY
    Product "1" --> "1..*" Ingredient : CONTAINS
    Product "1" --> "1..*" Category : BELONGS_TO
    class Product{
        +UUID id
        +String name
        +Float price
        +List dietary_tags
    }
    class Brand{
        +String name
        +String country
    }
    class Ingredient{
        +String name
        +Boolean is_allergen
    }
```

### 2. Technical Deep Dive (Extraction & Logic)
*Copy this block to explain the Python and Logic layers.*

```markdown
## 🛠️ Technical Deep Dive

### 1. The Miner (Structured Extraction)
We utilize **Pydantic** objects to enforce strict schema validation on LLM outputs. This ensures that messy catalog text is converted into clean, type-safe JSON before it ever touches the database.

**Tech:** `OpenAI GPT-4o-mini`, `Instructor`, `Pydantic`

```python
# Example: Enforcing schema constraints on raw text
class ProductSchema(BaseModel):
    brand: str
    dietary_tags: List[str] = Field(description="Tags like Gluten-Free, Vegan")
    ingredients: List[str]
    unit: str

# Result: Raw text "Oreo... contains wheat" automatically maps to:
# {"brand": "Oreo", "ingredients": ["Wheat"], "dietary_tags": ["Contains Wheat"]}
```

### 3. Installation & Usage
*Copy this block for the setup instructions.*

```markdown
## 💻 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/RetailGraph.git
   cd RetailGraph
```

### 4. Repository Structure & Roadmap
*Copy this block to finish the README.*

```markdown
## 📁 Repository Structure

```bash
RetailGraph/
├── data/
│   ├── raw/                 # Source CSVs
│   └── images/              # Downloaded product images
├── src/
│   ├── extraction/          # LLM Pydantic Models
│   ├── graph/               # Neo4j Connection & Cypher Builders
│   └── rag/                 # LangChain Query Logic
├── app.py                   # Streamlit Frontend
├── requirements.txt
└── README.md
```


