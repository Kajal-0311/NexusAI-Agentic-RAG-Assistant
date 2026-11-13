"""
Document Library Page: Upload, inspect, re-index, and delete research PDFs.
"""
import streamlit as st
from frontend.components.api_client import api_client


def render_document_library():
    st.title("📁 Document Library")
    st.caption("Upload research PDFs, manage chunk embeddings, and inspect per-page metadata.")

    # 1. Multi-PDF Upload Section
    with st.expander("⬆️ Upload New Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Select one or multiple PDF files (Max 25MB each)",
            type=["pdf"],
            accept_multiple_files=True,
            help="PDFs are validated, checked for duplicate hashes, split recursively into chunks with metadata preservation, and indexed into ChromaDB.",
        )

        if uploaded_files:
            if st.button("Index Selected Documents", type="primary"):
                with st.spinner("Processing documents, extracting pages, and computing embeddings..."):
                    file_tuples = [
                        ("files", (f.name, f.getvalue(), "application/pdf"))
                        for f in uploaded_files
                    ]
                    try:
                        resp = api_client.upload_documents(file_tuples)
                        st.success(resp.get("message", "Documents processed successfully."))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Upload error: {str(e)}")

    st.markdown("---")

    # 2. Document Search & Filter
    docs = api_client.get_documents()
    col_search, col_stats = st.columns([3, 1])

    with col_search:
        search_term = st.text_input(
            "🔍 Filter documents by filename", placeholder="Type a filename..."
        )
    with col_stats:
        st.markdown(f"**Total Indexed:** {len(docs)} document(s)")

    filtered_docs = (
        [d for d in docs if search_term.lower() in d["filename"].lower()]
        if search_term
        else docs
    )

    if not filtered_docs:
        st.info("No documents found matching your filter.")
        return

    # 3. Document Cards / Table
    for doc in filtered_docs:
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1.5])
            with c1:
                st.markdown(f"**📄 {doc['filename']}**")
                st.caption(f"ID: `{doc['document_id']}` | Uploaded: {doc['uploaded_at'][:19]}")
                if doc.get("is_scanned"):
                    st.warning("⚠️ Scanned PDF detected: minimal text was found. OCR is recommended.")
                if doc.get("error_message"):
                    st.error(doc["error_message"])

            with c2:
                st.metric("Pages", doc["total_pages"])
            with c3:
                st.metric("Chunks", doc["total_chunks"])
            with c4:
                size_kb = round(doc["file_size_bytes"] / 1024, 1)
                st.metric("Size", f"{size_kb} KB")
            with c5:
                # Actions: Reindex & Delete
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("🔄 Reindex", key=f"reindex_{doc['document_id']}", help="Re-chunk and re-embed"):
                        with st.spinner("Re-indexing..."):
                            api_client.reindex_document(doc["document_id"])
                            st.success("Re-indexed!")
                            st.rerun()
                with btn_col2:
                    if st.button("🗑️ Delete", key=f"del_{doc['document_id']}", help="Purge document and all vector records"):
                        api_client.delete_document(doc["document_id"])
                        st.warning("Deleted.")
                        st.rerun()

            st.markdown("<hr style='margin: 0.5rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
