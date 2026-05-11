"""LangGraph nodes: classify document type and generate structured summary."""

from app.state import AgentState
from app.llm_service import classify_doc_type, generate_topic_path, summarize_paper


def detect_doc_type(state: AgentState) -> AgentState:
    """Classify the paper type based on extracted metadata."""
    meta = state.get("parsed_metadata", {}) or {}
    title = meta.get("title", "")
    abstract = meta.get("abstract", "")
    keywords = meta.get("keywords", "")

    if not title:
        state["error"] = "Cannot classify: no title extracted."
        return state

    try:
        state["doc_type"] = classify_doc_type(title, abstract, keywords)
        state["topic_path"] = generate_topic_path(title, abstract, keywords)
        state["error"] = ""
    except Exception as e:
        state["error"] = f"Classification failed: {str(e)}"

    return state


def summarize_paper_node(state: AgentState) -> AgentState:
    """Generate a structured summary of the paper."""
    meta = state.get("parsed_metadata", {}) or {}
    title = meta.get("title", "")
    abstract = meta.get("abstract", "")
    sections = meta.get("sections", [])
    doc_type = state.get("doc_type", "method")
    level = state.get("summary_level", "standard")

    if not title:
        state["error"] = "Cannot summarize: no title extracted."
        return state

    try:
        state["summary_data"] = summarize_paper(title, abstract, sections, doc_type, level)
        state["error"] = ""
    except Exception as e:
        state["error"] = f"Summarization failed: {str(e)}"
        state["summary_data"] = None

    return state
