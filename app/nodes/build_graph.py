"""LangGraph node: build graph relationships for a paper."""

from app.state import AgentState
from app.services.graph_service import build_paper_graph


def build_graph(state: AgentState) -> AgentState:
    """Create graph nodes and edges: paper -> topic, paper -> doc_type, topic hierarchy."""
    paper_id = state.get("result", {}).get("paper_id", "")
    meta = state.get("parsed_metadata", {}) or {}
    title = meta.get("title", "")
    topic_path = state.get("topic_path", "")
    doc_type = state.get("doc_type", "method")

    if not paper_id:
        state["error"] = "Cannot build graph: no paper_id."
        return state

    try:
        build_paper_graph(paper_id, title, topic_path, doc_type)
        state["error"] = ""
    except Exception as e:
        state["error"] = f"Graph building failed: {str(e)}"

    return state
