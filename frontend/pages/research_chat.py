"""
Research Chat Page: Agentic multi-turn document investigation with citations and source transparency.
"""
import streamlit as st
from frontend.components.api_client import api_client


def render_research_chat():
    st.title("🔬 Research Chat")
    st.caption("Investigate documents using controlled Agentic RAG, exact citations, and multi-strategy retrieval.")

    # 1. Sidebar Session Controls & Retrieval Configuration
    docs = api_client.get_documents()
    sessions = api_client.get_sessions()

    # Active session state management
    if "current_session_id" not in st.session_state:
        if sessions:
            st.session_state["current_session_id"] = sessions[0]["session_id"]
        else:
            new_s = api_client.create_session("Initial Research Session")
            st.session_state["current_session_id"] = new_s.get("session_id", "default")

    # Session Selector in Sidebar
    st.sidebar.markdown("### 🗂️ Chat Sessions")
    session_options = {s["session_id"]: s["title"] for s in sessions} if sessions else {}

    if session_options:
        selected_session = st.sidebar.selectbox(
            "Select Session",
            options=list(session_options.keys()),
            format_func=lambda x: session_options.get(x, x),
            index=list(session_options.keys()).index(st.session_state["current_session_id"])
            if st.session_state["current_session_id"] in session_options
            else 0,
        )
        st.session_state["current_session_id"] = selected_session

    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("➕ New Chat", use_container_width=True):
            new_s = api_client.create_session("New Research Session")
            st.session_state["current_session_id"] = new_s.get("session_id")
            st.rerun()
    with col_btn2:
        if st.button("🧹 Clear", use_container_width=True, help="Clear message history in this session"):
            api_client.clear_session_messages(st.session_state["current_session_id"])
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Retrieval Controls")

    retrieval_mode = st.sidebar.selectbox(
        "Retrieval Strategy",
        options=["hybrid", "semantic", "keyword", "hybrid_rerank"],
        format_func=lambda x: {
            "hybrid": "Hybrid Search (RRF)",
            "semantic": "Semantic Search (Dense Vector)",
            "keyword": "BM25 Keyword Search",
            "hybrid_rerank": "Hybrid + Cross-Encoder Rerank",
        }.get(x, x),
        help="Select the information retrieval strategy used by the agent.",
    )

    top_k = st.sidebar.slider("Top-K Passages", min_value=1, max_value=10, value=5)

    # Document filter
    doc_options = {d["document_id"]: d["filename"] for d in docs}
    selected_doc_ids = None
    if doc_options:
        filter_choice = st.sidebar.multiselect(
            "Filter to Specific Documents",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options.get(x, x),
            placeholder="All documents (default)",
        )
        if filter_choice:
            selected_doc_ids = filter_choice

    # 2. Main Chat Area: Display Existing Message History
    session_data = api_client.get_session(st.session_state["current_session_id"])
    messages = session_data.get("messages", []) if session_data else []

    if not messages:
        st.info(
            "👋 Welcome to NexusAI Research Chat! Ask questions about your uploaded documents, "
            "request comparative summaries, or explore specific technical mechanisms."
        )

    for msg in messages:
        role = msg.get("role", "user")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))

            # Display citations if present in assistant response
            citations = msg.get("citations", [])
            if citations:
                st.markdown("**Citations & Sources:**")
                citation_pills = " ".join(
                    [
                        f"`Doc: {c.get('filename')} (Page {c.get('page_number')})`"
                        for c in citations
                    ]
                )
                st.markdown(citation_pills)

            # Expandable passages & agent trace for assistant replies
            passages = msg.get("retrieved_passages", [])
            traces = msg.get("step_traces", [])

            if passages or traces:
                with st.expander("🔍 View Retrieved Passages & Agent Reasoning"):
                    tab_passages, tab_trace = st.tabs(["📄 Evidence Passages", "🤖 Agent Workflow Trace"])

                    with tab_passages:
                        if not passages:
                            st.write("No passages retrieved for this response.")
                        for idx, p in enumerate(passages, start=1):
                            st.markdown(
                                f"**Passage {idx}** | *{p.get('filename')}* (Page {p.get('page_number')}) | "
                                f"Score: `{p.get('score', 0.0):.4f}` | Strategy: `{p.get('retrieval_strategy', 'unknown')}`"
                            )
                            st.text(p.get("content", ""))
                            st.markdown("---")

                    with tab_trace:
                        if not traces:
                            st.write("No agent trace recorded for this step.")
                        for t in traces:
                            st.markdown(
                                f"**Step {t.get('step_number')}: {t.get('step_name')}** ({t.get('latency_ms', 0)}ms)\n"
                                f"- *Action:* {t.get('action_taken')}\n"
                                f"- *Result:* {t.get('output_summary')}"
                            )

    # 3. Chat Input Box
    user_query = st.chat_input("Ask a question about your uploaded research documents...")
    if user_query:
        # Display user message immediately in UI
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Agent running: Classifying -> Contextual Rewriting -> Retrieving -> Verifying..."):
                try:
                    resp = api_client.send_chat_message(
                        query=user_query,
                        session_id=st.session_state["current_session_id"],
                        retrieval_mode=retrieval_mode,
                        top_k=top_k,
                        document_ids=selected_doc_ids,
                        use_reranker=retrieval_mode == "hybrid_rerank",
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate response: {str(e)}")
