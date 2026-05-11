"""LangGraph node: chunk text and index into Milvus."""

from app.state import AgentState
from app.services.milvus_service import (
    init_milvus,
    index_paper_chunks,
    index_paper_summary,
)


def index_milvus(state: AgentState) -> AgentState:
    """Chunk paper text, generate embeddings, and insert into Milvus collections."""
    meta = state.get("parsed_metadata", {}) or {}
    full_text = meta.get("full_text", "")
    topic_path = state.get("topic_path", "")
    doc_type = state.get("doc_type", "method")
    paper_id = state.get("result", {}).get("paper_id", "")
    summary_data = state.get("summary_data", {}) or {}
    summary_level = state.get("summary_level", "standard")

    if not paper_id or not full_text:
        state["error"] = "Cannot index: missing paper_id or text."
        return state

    try:
        init_milvus()
        chunk_count = index_paper_chunks(paper_id, full_text, topic_path=topic_path, doc_type=doc_type)
        summary_text = summary_data.get("paper_summary", meta.get("abstract", ""))
        index_paper_summary(paper_id, summary_text, level=summary_level, topic_path=topic_path)

        # Store chunk count in result
        state["result"]["chunks_indexed"] = chunk_count
        state["error"] = ""
    except Exception as e:
        state["error"] = f"Milvus indexing failed: {str(e)}"

    return state
