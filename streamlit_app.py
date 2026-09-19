"""Small chat UI for the FMR RAG API."""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="NBP FMR Research Chat",
    page_icon="FMR",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17324d;
        --muted: #62758a;
        --line: #dce6ef;
        --paper: #ffffff;
        --background: #f5f8fb;
        --blue: #1769aa;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--background);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    .stApp h1 {
        color: var(--ink);
        font-size: 2.15rem;
        letter-spacing: 0;
        margin-bottom: 0.35rem;
    }

    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stChatMessage"] p,
    .stApp label {
        color: var(--muted);
    }

    .stApp [data-testid="stMarkdownContainer"] strong,
    .stApp [data-testid="stMarkdownContainer"] h1,
    .stApp [data-testid="stMarkdownContainer"] h2,
    .stApp [data-testid="stMarkdownContainer"] h3 {
        color: var(--ink);
    }

    [data-testid="stChatMessage"] {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 10px;
        margin: 0.75rem 0;
        padding: 0.85rem 1rem;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: var(--ink);
    }

    [data-testid="stChatInput"] {
        border-color: #b9cddd;
    }

    [data-testid="stBottomBlockContainer"] {
        background: var(--background) !important;
    }

    .empty-state {
        max-width: 660px;
        margin: 8rem auto 5rem;
        padding: 2.1rem 2.4rem;
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(42, 72, 101, 0.08);
    }

    .empty-state h2 {
        color: var(--ink);
        font-size: 1.45rem;
        margin: 0 0 0.55rem;
    }

    .empty-state p {
        color: var(--muted);
        margin: 0;
    }

    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] textarea {
        color: var(--ink) !important;
        background: var(--paper) !important;
    }

    [data-testid="stChatInput"] input::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--muted) !important;
        opacity: 1;
    }

    [data-testid="stExpander"] {
        background: #f8fbfd;
        border: 1px solid var(--line);
        border-radius: 7px;
    }

    [data-testid="stMetric"] {
        background: #f8fbfd;
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 0.55rem 0.7rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--blue);
        font-size: 1.05rem;
    }

    .stApp [data-testid="stExpander"] summary,
    .stApp [data-testid="stExpander"] summary p {
        color: var(--ink) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NBP Fund Manager Report Chat")
st.caption("July 2026 | Conventional and Islamic funds | Answers grounded in supplied PDF reports")


def render_sources(sources: list[dict]) -> None:
    """Render deduplicated PDF evidence with clear verification metadata."""
    displayed_sources = set()
    unique_sources = []
    for source in sources:
        metadata = source.get("metadata", {})
        source_key = (metadata.get("source_file"), metadata.get("source_page"))
        if source_key in displayed_sources:
            continue
        displayed_sources.add(source_key)
        unique_sources.append(source)

    if not unique_sources:
        return

    st.markdown(f"### References ({len(unique_sources)})")
    st.caption("Verify important information against the cited PDF page.")
    for index, source in enumerate(unique_sources, start=1):
        metadata = source.get("metadata", {})
        category = metadata.get("category", "Unknown")
        report_period = f"{metadata.get('report_month', 'Unknown')} {metadata.get('report_year', '')}".strip()
        with st.container(border=True):
            st.markdown(f"**Source {index}: {metadata.get('source_file', 'Unknown PDF')}**")
            details = st.columns(4)
            details[0].metric("Category", category)
            details[1].metric("PDF page", metadata.get("source_page", "Unknown"))
            details[2].metric("Report period", report_period)
            section = metadata.get("matched_section", metadata.get("section", "General"))
            details[3].metric("Relevant section", section.replace("_", " ").title())
            with st.expander("View extracted evidence"):
                st.write(source.get("text", "No extracted text available."))

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <h2>Explore the July 2026 fund reports</h2>
            <p>Ask about a fund's risk profile, performance, asset allocation, or comparison.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

question = st.chat_input("Ask about a fund, return, risk profile, or comparison")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the FMR reports..."):
            try:
                result = requests.post(
                    f"{API_URL}/chat", json={"question": question}, timeout=300
                )
                result.raise_for_status()
                payload = result.json()
                st.markdown(payload["answer"])
                render_sources(payload.get("sources", []))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": payload["answer"],
                        "sources": payload.get("sources", []),
                    }
                )
            except requests.RequestException as exc:
                detail = ""
                if exc.response is not None:
                    try:
                        detail = exc.response.json().get("detail", "")
                    except ValueError:
                        detail = exc.response.text
                st.error(detail or f"Could not reach the API: {exc}")
