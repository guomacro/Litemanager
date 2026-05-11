"""LangGraph node: delete a paper and all associated data."""

import os

from app.state import AgentState
from app.services.metadata_service import (
    get_paper,
    soft_delete_paper,
    hard_delete_paper,
    delete_graph_for_paper,
)
from app.services.milvus_service import delete_paper_vectors


def delete_paper_node(state: AgentState) -> AgentState:
    """Execute full deletion: locate paper, remove vectors, graph, metadata, file."""
    paper_id = state.get("paper_id", "")
    hard = state.get("hard_delete", False)

    if not paper_id:
        state["error"] = "No paper_id provided for deletion."
        return state

    try:
        paper = get_paper(paper_id)
        if not paper:
            state["result"] = {"status": "not_found", "paper_id": paper_id}
            state["error"] = ""
            return state

        file_path = paper.get("file_path", "")

        if hard:
            # Hard delete: remove everything
            delete_paper_vectors(paper_id)
            delete_graph_for_paper(paper_id)
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            hard_delete_paper(paper_id)
            state["result"] = {"status": "hard_deleted", "paper_id": paper_id, "title": paper["title"]}
        else:
            # Soft delete: mark as deleted only
            soft_delete_paper(paper_id)
            state["result"] = {"status": "soft_deleted", "paper_id": paper_id, "title": paper["title"]}

        state["error"] = ""
    except Exception as e:
        state["error"] = f"Deletion failed: {str(e)}"

    return state
