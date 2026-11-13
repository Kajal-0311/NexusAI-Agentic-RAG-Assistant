"""
Settings & Telemetry Diagnostics Page for NexusAI.
"""
import streamlit as st
from frontend.components.api_client import api_client


def render_settings():
    st.title("⚙️ System Settings & Diagnostics")
    st.caption("Configure inference endpoints, chunking parameters, and verify service diagnostics.")

    current_settings = api_client.get_settings()
    health = api_client.get_health()

    # 1. Live Diagnostics
    st.subheader("System Diagnostic Health")
    c1, c2, c3 = st.columns(3)

    backend_ok = health is not None
    ollama_ok = bool(health and health.get("ollama_status", {}).get("is_running"))
    chroma_ok = bool(health and health.get("vector_store_status", {}).get("total_chunks", -1) >= 0)

    with c1:
        st.markdown(
            f"**FastAPI Backend:** {'🟢 Online' if backend_ok else '🔴 Unreachable'}"
        )
    with c2:
        st.markdown(
            f"**Ollama LLM Server:** {'🟢 Connected' if ollama_ok else '🟡 Fallback Mode'}"
        )
    with c3:
        st.markdown(
            f"**ChromaDB Vector Store:** {'🟢 Active' if chroma_ok else '🔴 Error'}"
        )

    st.markdown("---")

    # 2. Configuration Form
    with st.form("settings_form"):
        st.subheader("Inference & Models")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            ollama_url = st.text_input(
                "Ollama Base URL",
                value=current_settings.get("ollama_base_url", "http://localhost:11434"),
            )
            ollama_model = st.text_input(
                "Ollama LLM Model",
                value=current_settings.get("ollama_llm_model", "llama3"),
                help="Recommended: llama3, llama3.2, mistral",
            )
        with col_m2:
            embed_model = st.text_input(
                "Embedding Model",
                value=current_settings.get("ollama_embed_model", "nomic-embed-text"),
                disabled=True,
                help="Standardized nomic-embed-text vector embedding model.",
            )
            use_reranker = st.checkbox(
                "Enable Cross-Encoder Reranking by Default",
                value=current_settings.get("use_reranker", False),
            )

        st.subheader("Document Chunking & Processing")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            chunk_size = st.number_input(
                "Chunk Size (Characters)",
                min_value=100,
                max_value=2000,
                value=current_settings.get("chunk_size", 500),
                step=50,
            )
        with col_c2:
            chunk_overlap = st.number_input(
                "Chunk Overlap (Characters)",
                min_value=0,
                max_value=500,
                value=current_settings.get("chunk_overlap", 100),
                step=25,
            )

        st.subheader("Default Retrieval Parameters")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            retrieval_mode = st.selectbox(
                "Default Retrieval Strategy",
                options=["hybrid", "semantic", "keyword"],
                index=["hybrid", "semantic", "keyword"].index(
                    current_settings.get("default_retrieval_mode", "hybrid")
                )
                if current_settings.get("default_retrieval_mode") in ["hybrid", "semantic", "keyword"]
                else 0,
            )
        with col_r2:
            top_k = st.number_input(
                "Default Top-K",
                min_value=1,
                max_value=15,
                value=current_settings.get("default_top_k", 5),
            )
        with col_r3:
            rrf_k = st.number_input(
                "RRF Constant k",
                min_value=1,
                max_value=120,
                value=current_settings.get("rrf_k", 60),
                disabled=True,
            )

        submitted = st.form_submit_button("Save Settings", type="primary")
        if submitted:
            update_payload = {
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "default_retrieval_mode": retrieval_mode,
                "default_top_k": int(top_k),
                "use_reranker": use_reranker,
                "ollama_model": ollama_model,
            }
            try:
                api_client.update_settings(update_payload)
                st.success("Configuration updated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to update settings: {e}")
