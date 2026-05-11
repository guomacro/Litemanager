"""LangGraph node: persist metadata to SQLite."""

import shutil
from pathlib import Path

from app.state import AgentState
from app.config import PAPER_DIR
from app.services.metadata_service import insert_paper, save_summary


def persist_metadata(state: AgentState) -> AgentState:
    """Write paper metadata and summary to SQLite, copy PDF to data directory."""
    meta = state.get("parsed_metadata", {}) or {}
    title = meta.get("title", "")
    abstract = meta.get("abstract", "")
    keywords = meta.get("keywords", "")
    doc_type = state.get("doc_type", "method")
    topic_path = state.get("topic_path", "")
    summary_data = state.get("summary_data", {}) or {}
    summary_level = state.get("summary_level", "standard")
    pdf_path = state.get("pdf_path", "")

    if not title:
        state["error"] = "Cannot persist: no title."
        return state

    try:
        # Copy PDF to data directory
        dest_path = ""
        file_name = ""
        if pdf_path:
            src = Path(pdf_path)
            file_name = src.name
            dest = PAPER_DIR / src.name
            if dest.exists():
                dest = PAPER_DIR / f"{src.stem}_{src.stat().st_size}{src.suffix}"
            shutil.copy2(src, dest)
            dest_path = str(dest)

        # Insert paper record
        paper_id = insert_paper(
            title=title,
            abstract=abstract,
            keywords=keywords,
            doc_type=doc_type,
            topic_path=topic_path,
            file_path=dest_path,
            file_name=file_name,
        )

        # Save summary
        save_summary(paper_id, summary_level, summary_data)

        state["result"] = {
            "paper_id": paper_id,
            "title": title,
            "doc_type": doc_type,
            "topic_path": topic_path,
            "abstract": abstract,
            "keywords": keywords,
            "summary": summary_data,
            "file_path": dest_path,
        }
        state["error"] = ""
    except Exception as e:
        state["error"] = f"Metadata persistence failed: {str(e)}"

    return state
