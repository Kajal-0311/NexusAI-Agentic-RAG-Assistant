"""
Dashboard Page: Overview of collection size, model health, and quick actions.
"""
import streamlit as st
from frontend.components.api_client import api_client


def render_dashboard():
    st.title("NexusAI: System Overview")
    st.caption("Agentic Retrieval-Augmented Generation & Multi-Strategy Research Assistant")

    # Fetch status data
    health = api_client.get_health()
    docs = api_client.get_documents()
    sessions = api_client.get_sessions()

    # Top Metric KPI Row
    col1, col2, col3, col4 = st.columns(4)

    total_docs = len(docs)
    total_chunks = sum(d.get("total_chunks", 0) for d in docs)
    total_sessions = len(sessions)
    llm_connected = bool(health and health.get("ollama_status", {}).get("is_running"))

    with col1:
        st.metric(label="Total Documents", value=total_docs)
    with col2:
        st.metric(label="Indexed Chunks", value=f"{total_chunks:,}")
    with col3:
        st.metric(label="Chat Sessions", value=total_sessions)
    with col4:
        st.metric(
            label="Ollama LLM",
            value="Online" if llm_connected else "Fallback Mode",
            delta="Llama 3 Active" if llm_connected else "Local Engine",
            delta_color="normal" if llm_connected else "off",
        )

    st.markdown("---")

    # System Status & Quick Actions
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("System Architecture & Health")
        if health:
            v_stats = health.get("vector_store_status", {})
            st.markdown(
                f"""
                - **Vector Database:** ChromaDB (`{v_stats.get('collection_name', 'nexus_knowledge_base')}`) with **{v_stats.get('total_chunks', 0)}** active vectors.
                - **Primary Embedding Model:** `{health.get('ollama_status', {}).get('target_model', 'nomic-embed-text')}`
                - **Relational Metadata Store:** SQLite WAL (`nexus_ai.db`) - **Connected**
                - **Retrieval Engine:** Semantic Cosine + BM25 Okapi + RRF (Reciprocal Rank Fusion) + Cross-Encoder
                """
            )
            if not llm_connected:
                st.info(
                    "💡 **Ollama Notice:** Ollama is not currently detected on `localhost:11434`. "
                    "The system has seamlessly engaged its built-in deterministic grounding engine, allowing complete end-to-end testing, retrieval, citations, and evaluation!"
                )
        else:
            st.warning("Backend API is currently unreachable. Please make sure FastAPI backend is running.")

        st.subheader("Recent Research Documents")
        if docs:
            doc_table = [
                {
                    "Filename": d["filename"],
                    "Pages": d["total_pages"],
                    "Chunks": d["total_chunks"],
                    "Size (KB)": round(d["file_size_bytes"] / 1024, 1),
                    "Status": "Processed" if d["is_processed"] else "Pending",
                }
                for d in docs[:5]
            ]
            st.dataframe(doc_table, use_container_width=True)
        else:
            st.info("No documents uploaded yet. Go to the **Document Library** to index research PDFs.")

    with col_right:
        st.subheader("Quick Actions")
        st.markdown(
            """
            Explore the flagship modules of NexusAI:
            """
        )
        if st.button("📁 Open Document Library", use_container_width=True):
            st.session_state["nav_page"] = "Document Library"
            st.rerun()

        if st.button("🔬 Start Research Chat", use_container_width=True):
            st.session_state["nav_page"] = "Research Chat"
            st.rerun()

        if st.button("📈 Run Benchmark Evaluation", use_container_width=True):
            st.session_state["nav_page"] = "Evaluation Dashboard"
            st.rerun()

        st.markdown("---")
        st.subheader("Retrieval Strategy Highlights")
        st.markdown(
            """
            - **Semantic:** Dense vector embeddings capture semantic meaning.
            - **BM25 Keyword:** Inverted index captures exact acronyms, equations, and technical identifiers.
            - **Hybrid RRF:** Fuses dense and sparse rankings with zero score calibration skew.
            - **Cross-Encoder Rerank:** Joint query-document attention maximizes Precision@K.
            """
        )
