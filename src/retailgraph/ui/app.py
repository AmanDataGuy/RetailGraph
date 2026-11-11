"""
# app.py - Streamlit chat UI for the RetailGraph agent

- Sends the user's question to the FastAPI backend (POST {API_URL}/v1/query) and shows answer + product cards.
- Run: `streamlit run src/retailgraph/ui/app.py` (API_URL env var, default http://127.0.0.1:8000).
"""

import os
from html import escape

import requests
import streamlit as st

st.set_page_config(
    page_title="RetailGraph",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# docker-compose sets API_URL=http://api:8000; locally the API runs beside the UI.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# White, minimal theme.
st.markdown("""
<style>
    /* Force dark text: Streamlit's dark-mode text color washes out on white */
    .stApp { background-color: #ffffff; }
    .stApp, .stApp p, .stApp span, .stApp li, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    [data-testid="stMarkdownContainer"] { color: #1a1a1a; }

    /* Sidebar background + text (Streamlit renders this as a separate container) */
    [data-testid="stSidebar"] {
        background-color: #fafafa;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] * { color: #1a1a1a; }
    [data-testid="stSidebar"] button {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        color: #1a1a1a;
    }

    /* Product card */
    .product-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
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
        color: #1a1a1a;
        margin: 8px 0 4px 0;
        line-height: 1.3;
        min-height: 2.6em;
    }

    /* Price badge */
    .price-badge {
        background: #e8f5e9;
        color: #2e7d32;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.9rem;
        font-weight: 700;
        display: inline-block;
        margin: 4px 0;
    }

    /* Tag pill */
    .tag-pill {
        background: #e3f2fd;
        color: #1565c0;
        border-radius: 20px;
        padding: 2px 8px;
        font-size: 0.7rem;
        display: inline-block;
        margin: 2px 2px 2px 0;
    }

    /* Category label */
    .category-label {
        color: #757575;
        font-size: 0.75rem;
        margin: 4px 0;
    }

    /* Answer box */
    .answer-box {
        background: #f5f7fa;
        border-left: 4px solid #4f8ef7;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 16px 0;
        color: #1a1a1a;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Sidebar stat */
    .stat-box {
        background: #f5f7fa;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.8rem;
        color: #333333;
    }
    .stat-label {
        color: #757575;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Cypher block */
    .cypher-block {
        background: #f5f7fa;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 10px;
        font-family: monospace;
        font-size: 0.72rem;
        color: #0b5394;
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
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        border-radius: 8px;
        color: #1a1a1a;
        font-size: 1rem;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# Backend health, cached 60s because /v1/health makes a real Groq call and Streamlit reruns often.
@st.cache_data(ttl=60)
def check_health() -> dict:
    try:
        r = requests.get(f"{API_URL}/v1/health", timeout=10)
        return r.json()
    except Exception:
        return {"status": "unreachable", "neo4j": False, "qdrant": False, "groq": False}


# Send the question to the agent API; show an error and return None on failure.
def run_query(query: str) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/v1/query", json={"query": query}, timeout=30)
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI. Make sure `uvicorn retailgraph.api.main:app --reload --port 8000` is running.")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


# Draw one product card: image, name, price, category/brand, up to 3 tags.
def render_product_card(product: dict):
    image_url = product.get("image_url")
    name      = product.get("item_name", "Unknown Product")
    price     = product.get("price")
    category  = product.get("category", "")
    brand     = product.get("brand", "")
    tags      = product.get("dietary_tags") or []

    if image_url:
        try:
            st.image(image_url, use_column_width=True)
        except Exception:
            st.markdown("🖼️", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='background:#f0f0f0;border-radius:8px;height:120px;"
            "display:flex;align-items:center;justify-content:center;"
            "color:#aaaaaa;font-size:2rem;'>🛒</div>",
            unsafe_allow_html=True
        )

    st.markdown(f"<div class='product-name'>{escape(str(name))}</div>", unsafe_allow_html=True)

    if price is not None:
        st.markdown(f"<div class='price-badge'>${price:.2f}</div>", unsafe_allow_html=True)

    meta = [x for x in (category, brand) if x]
    if meta:
        st.markdown(f"<div class='category-label'>{escape(' · '.join(meta))}</div>", unsafe_allow_html=True)

    if tags:
        tag_html = "".join(f"<span class='tag-pill'>🌱 {escape(str(t))}</span>" for t in tags[:3])
        st.markdown(tag_html, unsafe_allow_html=True)


# Show the agent internals (intent, route, result count, latency, Cypher) in the sidebar.
def render_sidebar(response: dict):
    with st.sidebar:
        st.markdown("### 🔍 Agent Internals")

        intent = response.get("intent", "—")
        route  = response.get("route", "—")
        count  = response.get("result_count", 0)
        total  = response.get("total_matches")
        lat    = response.get("latency_ms", 0)
        results_label = f"{count} of {total} shown" if total and total > count else f"{count} products"

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
            <div>{escape(str(intent_icon))}</div>
        </div>
        <div class='stat-box'>
            <div class='stat-label'>Route</div>
            <div>{escape(str(route_icon))}</div>
        </div>
        <div class='stat-box'>
            <div class='stat-label'>Results</div>
            <div>{results_label}</div>
        </div>
        <div class='stat-box'>
            <div class='stat-label'>Latency</div>
            <div>{lat:.0f} ms</div>
        </div>
        """, unsafe_allow_html=True)

        cypher = response.get("cypher_used")
        if cypher and cypher != "GraphRAG (Qdrant semantic + Neo4j constraints)":
            st.markdown("**Cypher Query**")
            st.markdown(f"<div class='cypher-block'>{escape(cypher)}</div>", unsafe_allow_html=True)
        elif cypher:
            st.markdown("**Search Method**")
            st.markdown(f"<div class='cypher-block'>{escape(cypher)}</div>", unsafe_allow_html=True)


# Page layout: sidebar (health + example queries), query box, answer and results grid.
def main():
    with st.sidebar:
        st.markdown("## 🛒 RetailGraph")
        st.markdown("*Multimodal Knowledge Graph*")
        st.divider()

        health = check_health()

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

    st.markdown("# 🛒 RetailGraph")
    st.markdown("Ask anything about grocery products — powered by GraphRAG + LangGraph")
    st.divider()

    query = st.text_input(
        label="Query",
        placeholder="e.g. show me vegan snacks under $10 with no nuts",
        key="query_input",
        label_visibility="collapsed",
    )

    # Set when an example button was clicked
    run_now = st.session_state.pop("run_query", False)

    if (query and st.button("🔍 Search", type="primary")) or (run_now and query):
        with st.spinner("Running agent..."):
            response = run_query(query)

        if response:
            render_sidebar(response)

            answer = response.get("answer", "")
            if answer:
                st.markdown(
                    f"<div class='answer-box'>💬 {escape(answer)}</div>",
                    unsafe_allow_html=True
                )

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
        # Landing state: example questions
        st.markdown("### What can you ask?")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(r"""
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