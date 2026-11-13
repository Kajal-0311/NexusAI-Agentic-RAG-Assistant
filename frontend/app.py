"""
NexusAI - Agentic RAG Research Assistant
Main Streamlit Application Entrypoint.
"""
from pathlib import Path
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="NexusAI - Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS Stylesheet
css_path = Path(__file__).parent / "styles" / "main.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Import sub-pages
from frontend.components.api_client import api_client
from frontend.pages.dashboard import render_dashboard
from frontend.pages.document_library import render_document_library
from frontend.pages.evaluation import render_evaluation
from frontend.pages.research_chat import render_research_chat
from frontend.pages.settings import render_settings

# Navigation State
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Dashboard"

# Sidebar Branding & Navigation
with st.sidebar:
    st.markdown("## 🔬 **NexusAI**")
    st.caption("Agentic RAG Research Assistant")

    page_selection = st.radio(
        "Navigation",
        options=[
            "Dashboard",
            "Document Library",
            "Research Chat",
            "Evaluation Dashboard",
            "Settings",
        ],
        index=[
            "Dashboard",
            "Document Library",
            "Research Chat",
            "Evaluation Dashboard",
            "Settings",
        ].index(st.session_state["nav_page"]),
        label_visibility="collapsed",
    )
    st.session_state["nav_page"] = page_selection

    st.markdown("---")

    # Live Health Pill
    health = api_client.get_health()
    if health:
        is_ollama = health.get("ollama_status", {}).get("is_running")
        status_text = "🟢 Online" if is_ollama else "🟡 Local Fallback"
        st.caption(f"**Inference:** {status_text}")
        st.caption(f"**Backend:** 🟢 Connected (v1.0.0)")
    else:
        st.caption("**Backend:** 🔴 Offline (Start FastAPI)")

    st.markdown("---")
    st.markdown(
        "<small style='color: #94a3b8;'>NexusAI v1.0.0 • B.Tech AI/ML Portfolio Project</small>",
        unsafe_allow_html=True,
    )

# Page Router
if st.session_state["nav_page"] == "Dashboard":
    render_dashboard()
elif st.session_state["nav_page"] == "Document Library":
    render_document_library()
elif st.session_state["nav_page"] == "Research Chat":
    render_research_chat()
elif st.session_state["nav_page"] == "Evaluation Dashboard":
    render_evaluation()
elif st.session_state["nav_page"] == "Settings":
    render_settings()
