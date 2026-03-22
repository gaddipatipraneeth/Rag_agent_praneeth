import os,sys
sys.path.append(r'C:\Users\gaddi\Downloads\deep-learning-rag')  # Adjust the path as needed
from rag_agent.agent.state import AgentState
from langchain_core.messages import HumanMessage
from rag_agent.agent.nodes import query_rewrite_node, retrieval_node, generation_node
from rag_agent.vectorstore.store import VectorStoreManager
from rag_agent.agent.state import ChunkMetadata, DocumentChunk


def sample_chunk() -> DocumentChunk:
    """A single valid DocumentChunk for use across tests."""
    metadata = ChunkMetadata(
        topic="LSTM",
        difficulty="intermediate",
        type="concept_explanation",
        source="test_lstm.md",
        related_topics=["RNN", "vanishing_gradient"],
        is_bonus=False,
    )
    return DocumentChunk(
        chunk_id=VectorStoreManager.generate_chunk_id("test_lstm.md", "test content"),
        chunk_text=(
            "Long Short-Term Memory networks solve the vanishing gradient problem "
            "through gated mechanisms: the forget gate, input gate, and output gate. "
            "These gates control information flow through the cell state, allowing "
            "the network to maintain relevant information across long sequences."
        ),
        metadata=metadata,
    )
def test_agent_end_to_end(chunk, test_settings):
    # 1. Ingest a sample chunk
    store = VectorStoreManager(settings=test_settings)
    store.ingest([chunk])

    # # 2. Simulate a user query
    user_query = "How do LSTMs remember information?"
    state = AgentState()
    state = AgentState(messages=[HumanMessage(content=user_query)])

    # 3. Rewrite the query
    state_updates = query_rewrite_node(state)
    state["original_query"] = state_updates["original_query"]
    state["rewritten_query"] = state_updates["rewritten_query"]

    # 4. Retrieve relevant chunks
    retrieval_updates = retrieval_node(state)
    state["retrieved_chunks"] = retrieval_updates["retrieved_chunks"]
    state["no_context_found"] = retrieval_updates["no_context_found"]

    # 5. Generate a response
    generation_updates = generation_node(state)
    final_response = generation_updates["final_response"]

    # 6. Assert the response is not a hallucination guard and contains expected info
    assert not final_response.no_context_found
    assert "LSTM" in final_response.answer or "Long Short-Term Memory" in final_response.answer


if __name__ == "__main__":
    # Set up test settings and sample chunk as in your pytest fixtures
    from rag_agent.config import get_settings
    test_settings = get_settings()
    chunk = sample_chunk()
    test_agent_end_to_end(chunk, test_settings)
    print("End-to-end agent workflow test completed successfully.")