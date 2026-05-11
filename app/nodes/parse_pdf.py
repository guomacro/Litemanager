"""LangGraph node: parse PDF and extract metadata."""

from app.state import AgentState
from app.services.pdf_service import extract_text_from_pdf, extract_metadata_from_text


def parse_pdf(state: AgentState) -> AgentState:
    """Extract full text and metadata (title, abstract, keywords, sections) from a PDF."""
    pdf_path = state.get("pdf_path", "")

    if not pdf_path:
        state["error"] = "No PDF path provided."
        return state

    try:
        text = extract_text_from_pdf(pdf_path)
        metadata = extract_metadata_from_text(text)
        state["parsed_metadata"] = metadata
        state["error"] = ""
    except Exception as e:
        state["error"] = f"PDF parsing failed: {str(e)}"
        state["parsed_metadata"] = None

    return state
