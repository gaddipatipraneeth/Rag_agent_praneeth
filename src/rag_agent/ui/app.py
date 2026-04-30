"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": "default-session",  # LangGraph conversation thread
        "topic_filter": "All",
        "difficulty_filter": "All",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of PDF and Markdown files. Displays
    ingestion results (chunks added, duplicates skipped, errors).
    Updates the ingested documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True
    )

    if st.sidebar.button("Ingest Documents", disabled=not uploaded_files):
        with st.sidebar.spinner("Processing documents..."):
            # Save uploaded files to temp directory
            temp_dir = Path(tempfile.mkdtemp())
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)

            # Chunk files
            chunks = chunker.chunk_files(file_paths)

            # Ingest chunks
            result = store.ingest(chunks)

            # Display result
            if result.errors:
                st.sidebar.error(f"Errors during ingestion: {result.errors}")
            else:
                st.sidebar.success(f"✅ {result.ingested} chunks added, {result.skipped} duplicates skipped")

            # Refresh ingested documents list
            st.session_state.ingested_documents = store.list_documents()

    # Render ingested documents list
    if st.session_state.ingested_documents:
        st.sidebar.subheader("Ingested Documents")
        for doc in st.session_state.ingested_documents:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(f"📄 {doc['source']}")
                st.caption(f"Topic: {doc.get('topic', 'N/A')} | Chunks: {doc['chunk_count']}")
            with col2:
                if st.button("🗑", key=f"delete_{doc['source']}"):
                    deleted_count = store.delete_document(doc['source'])
                    st.success(f"Deleted {deleted_count} chunks from {doc['source']}")
                    st.session_state.ingested_documents = store.list_documents()
                    st.rerun()
    else:
        st.sidebar.info("Upload .pdf or .md files to populate the corpus.")


def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """
    stats = store.get_collection_stats()
    st.sidebar.metric("Total Chunks", stats["total_chunks"])
    st.sidebar.write("Topics:", ", ".join(stats["topics"]))
    if stats["bonus_topics_present"]:
        st.sidebar.success("✅ Bonus topics present")
    else:
        st.sidebar.warning("⚠️ No bonus topics yet")


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    docs = store.list_documents()
    if not docs:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    doc_sources = [doc["source"] for doc in docs]
    selected_source = st.selectbox("Select document", options=doc_sources, key="selected_document")

    if selected_source:
        chunks = store.get_document_chunks(selected_source)
        with st.container(height=600):
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}**")
                st.caption(f"Topic: {chunk.metadata.topic} | Difficulty: {chunk.metadata.difficulty} | Type: {chunk.metadata.type}")
                st.write(chunk.chunk_text)
                if hasattr(chunk, 'score') and chunk.score is not None:
                    st.caption(f"Similarity Score: {chunk.score:.3f}")
                st.divider()


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(graph,store) -> None:
    """
    Render the chat interface in the right column.

    Supports multi-turn conversation with the LangGraph agent.
    Displays source citations with every response.
    Shows a clear "no relevant context" indicator when the
    hallucination guard fires.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph agent from get_compiled_graph().
    """
    st.subheader("💬 Interview Prep Chat")

    # Filters
    docs = store.list_documents()
    topics = list(set(doc.get("topic") for doc in docs if doc.get("topic")))
    difficulties = ["beginner", "intermediate", "advanced"]  # assuming standard difficulties

    col_topic, col_diff = st.columns(2)
    with col_topic:
        selected_topic = st.selectbox("Topic Filter", options=["All"] + topics, key="topic_filter")
    with col_diff:
        selected_difficulty = st.selectbox("Difficulty Filter", options=["All"] + difficulties, key="difficulty_filter")

    # Chat history display
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    # Chat input
    query = st.chat_input("Ask about a deep learning topic...")

    if query:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Build filters
        filters = {}
        if st.session_state.topic_filter != "All":
            filters["topic"] = st.session_state.topic_filter
        if st.session_state.difficulty_filter != "All":
            filters["difficulty"] = st.session_state.difficulty_filter

        # Call the graph
        input_data = {"messages": [HumanMessage(content=query)], "filters": filters}
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        result = graph.invoke(input_data, config=config)

        # Append assistant message
        response = result["final_response"]
        sources = result.get("sources", [])
        no_context_found = result.get("no_context_found", False)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "sources": sources,
            "no_context_found": no_context_found
        })

        st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Populate initial ingested documents
    if not st.session_state.ingested_documents:
        st.session_state.ingested_documents = store.list_documents()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph, store)


if __name__ == "__main__":
    main()
