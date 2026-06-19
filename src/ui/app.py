"""
RetailGraph — Streamlit UI
A conversational interface for the RetailGraph knowledge graph agent.
Calls the FastAPI backend at localhost:8000.
"""

import streamlit as st
import requests
import time

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailGraph",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Product card */
    .product-card {
        background: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
        transition: border-color 0.2s;
    }
    .product-card:hover { border-color: #4f8ef7; }

    /* Product name */
    .product-name {
        font-size: 0.85rem;
        font-weight: 600;
        color: #e8eaf6;
        margin: 8px 0 4px 0;
        line-height: 1.3;
        min-height: 2.6em;
    }

    /* Price badge */
    .price-badge {
        background: #1a3a1a;
        color: #4caf50;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.9rem;
        font-weight: 700;
        display: inline-block;
        margin: 4px 0;
    }

    /* Tag pill */
    .tag-pill {
        background: #1a2a3a;
        color: #64b5f6;
        border-radius: 20px;
        padding: 2px 8px;
        font-size: 0.7rem;
        display: inline-block;
        margin: 2px 2px 2px 0;
    }

    /* Category label */
    .category-label {
        color: #9e9e9e;
        font-size: 0.75rem;
        margin: 4px 0;
    }

    /* Answer box */
    .answer-box {
        background: #1e2130;
        border-left: 4px solid #4f8ef7;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 16px 0;
        color: #e8eaf6;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Sidebar stat */
    .stat-box {
        background: #1e2130;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.8rem;
        color: #b0b8d4;
    }
    .stat-label {
        color: #9e9e9e;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Cypher block */
    .cypher-block {
        background: #0d1117;
        border: 1px solid #2e3250;
        border-radius: 6px;
        padding: 10px;
        font-family: monospace;
        font-size: 0.72rem;
        color: #79c0ff;
        white-space: pre-wrap;
        word-break: break-word;
        margin-top: 4px;
    }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Input box */
    .stTextInput > div > div > input {
        background-color: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 8px;
        color: #e8eaf6;
        font-size: 1rem;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def check_health() -> dict:
    try:
        r = requests.get(f"{API_URL}/v1/health", timeout=10)
        return r.json()
    except Exception:
        return {"status": "unreachable", "neo4j": False, "qdrant": False, "groq": False}

def run_query(query: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/v1/query",
            json={"query": query},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API error {r.status_code}: {r.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI. Make sure `uvicorn src.api.main:app --reload --port 8000` is running.")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def render_product_card(product: dict):
    """Render a single product card with image."""
    image_url = product.get("image_url")
    name      = product.get("item_name", "Unknown Product")
    price     = product.get("price")
    category  = product.get("category", "")
    brand     = product.get("brand", "")
    tags      = product.get("dietary_tags") or []

    # Image
    if image_url:
        try:
            st.image(image_url, use_column_width=True)
        except Exception:
            st.markdown("🖼️", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='background:#2a2a3a;border-radius:8px;height:120px;"
            "display:flex;align-items:center;justify-content:center;"
            "color:#555;font-size:2rem;'>🛒</div>",
            unsafe_allow_html=True
        )

    # Name
    st.markdown(f"<div class='product-name'>{name}</div>", unsafe_allow_html=True)

    # Price
    if price is not None:
        st.markdown(f"<div class='price-badge'>${price:.2f}</div>", unsafe_allow_html=True)

    # Category + Brand
    meta = []
    if category:
        meta.append(category)
    if brand:
        meta.append(brand)
    if meta:
        st.markdown(f"<div class='category-label'>{' · '.join(meta)}</div>", unsafe_allow_html=True)

    # Dietary tags
    if tags:
        tag_html = "".join(f"<span class='tag-pill'>🌱 {t}</span>" for t in tags[:3])
        st.markdown(tag_html, unsafe_allow_html=True)


def render_sidebar(response: dict):
    """Render the agent internals panel in the sidebar."""
    with st.sidebar:
        st.markdown("### 🔍 Agent Internals")

        # Status indicators
        intent = response.get("intent", "—")
        route  = response.get("route", "—")
        count  = response.get("result_count", 0)
        lat    = response.get("latency_ms", 0)

        route_icon = {
            "cypher":    "🗄️ Cypher",
            "graphrag":  "🔮 GraphRAG",
            "analytics": "📊 Analytics",
        }.get(route, route)

        intent_icon = {
            "filter":    "🔽 Filter",
            "semantic":  "🔍 Semantic",
            "hybrid":    "⚡ Hybrid",
            "lookup":    "🔎 Lookup",
            "analytics": "📊 Analytics",
        }.get(intent, intent)

        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-label'>Intent</div>
            <div>{intent_icon}</div>
        </div>
        <div class='stat-box'>
            <div class='stat-label'>Route</div>
            <div>{route_icon}</div>
        </div>
        <div class='stat-box'>
            <div class='stat-label'>Results</div>
            <div>{count} products</div>
        </div>
        <div class='stat-box'>
            <div class='stat-label'>Latency</div>
            <div>{lat:.0f} ms</div>
        </div>
        """, unsafe_allow_html=True)

        # Cypher query
        cypher = response.get("cypher_used")
        if cypher and cypher != "GraphRAG (Qdrant semantic + Neo4j constraints)":
            st.markdown("**Cypher Query**")
            st.markdown(f"<div class='cypher-block'>{cypher}</div>", unsafe_allow_html=True)
        elif cypher:
            st.markdown("**Search Method**")
            st.markdown(f"<div class='cypher-block'>{cypher}</div>", unsafe_allow_html=True)


# ── Main app ───────────────────────────────────────────────────────────────

def main():
    # ── Sidebar header ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🛒 RetailGraph")
        st.markdown("*Multimodal Knowledge Graph*")
        st.divider()

        # Health check
        health = check_health()
        status = health.get("status", "unreachable")

        col1, col2, col3 = st.columns(3)
        col1.metric("Neo4j",  "✅" if health.get("neo4j")  else "❌")
        col2.metric("Qdrant", "✅" if health.get("qdrant") else "❌")
        col3.metric("Groq",   "✅" if health.get("groq")   else "❌")

        st.divider()
        st.markdown("**Example queries**")
        examples = [
            "vegan snacks under $10",
            "something similar to sriracha",
            "gluten-free beverages no nuts",
            "which category has most products",
            "show me organic coffee",
            "keto snacks under $5",
        ]
        for ex in examples:
            if st.button(f"↗ {ex}", key=ex, use_container_width=True):
                st.session_state["query_input"] = ex
                st.session_state["run_query"]   = True

        st.divider()
        st.markdown(
            "<div style='color:#555;font-size:0.7rem;'>"
            "2,160 products · Neo4j AuraDB · Qdrant Cloud · Groq LPU"
            "</div>",
            unsafe_allow_html=True
        )

    # ── Main area ──────────────────────────────────────────────────────────
    st.markdown("# 🛒 RetailGraph")
    st.markdown("Ask anything about grocery products — powered by GraphRAG + LangGraph")
    st.divider()

    # Query input
    query = st.text_input(
        label="Query",
        placeholder="e.g. show me vegan snacks under $10 with no nuts",
        key="query_input",
        label_visibility="collapsed",
    )

    # Check if example button was clicked
    run_now = st.session_state.pop("run_query", False)

    if (query and st.button("🔍 Search", type="primary")) or (run_now and query):
        with st.spinner("Running agent..."):
            response = run_query(query)

        if response:
            # Render sidebar internals
            render_sidebar(response)

            # Answer box
            answer = response.get("answer", "")
            if answer:
                st.markdown(
                    f"<div class='answer-box'>💬 {answer}</div>",
                    unsafe_allow_html=True
                )

            # Results grid
            results = response.get("results", [])
            if results:
                st.markdown(f"### Products ({len(results)} found)")
                cols = st.columns(3)
                for i, product in enumerate(results):
                    with cols[i % 3]:
                        with st.container():
                            render_product_card(product)
                            st.divider()
            else:
                st.info("No products found. Try a different query.")

    elif not query:
        # Landing state — show stats
        st.markdown("### What can you ask?")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            **🔽 Filter queries**
            - vegan snacks under \$10
            - gluten-free beverages
            - organic coffee under \$15
            - keto snacks no nuts
            """)
        with c2:
            st.markdown("""
            **🔮 Semantic queries**
            - something like sriracha
            - similar to protein bars
            - hot sauce alternatives
            - find me kombucha style drinks
            """)
        with c3:
            st.markdown("""
            **📊 Analytics queries**
            - which category has most products
            - top brands in the graph
            - most common dietary tags
            - average price by category
            """)


if __name__ == "__main__":
    main()