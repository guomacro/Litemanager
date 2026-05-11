"""LangGraph workflow definition for the Literature Management Agent.

Workflows:
- import:  parse_pdf -> detect_doc_type -> summarize -> persist_metadata -> build_graph -> index_milvus
- delete:  delete_paper
- search:   search_papers -> generate_answer
- summarize: detect_doc_type -> summarize -> persist_metadata
"""

from langgraph.graph import StateGraph, END

from app.state import AgentState
from app.nodes.route_intent import route_intent, route_by_intent
from app.nodes.parse_pdf import parse_pdf
from app.nodes.summarize import detect_doc_type, summarize_paper_node
from app.nodes.build_graph import build_graph
from app.nodes.index_milvus import index_milvus
from app.nodes.persist_metadata import persist_metadata
from app.nodes.delete_paper import delete_paper_node
from app.nodes.search_papers import search_papers_node, generate_answer_node


def build_import_graph() -> StateGraph:
    """Build the import workflow subgraph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("parse_pdf", parse_pdf)
    workflow.add_node("detect_doc_type", detect_doc_type)
    workflow.add_node("summarize", summarize_paper_node)
    workflow.add_node("persist_metadata", persist_metadata)
    workflow.add_node("build_graph", build_graph)
    workflow.add_node("index_milvus", index_milvus)

    workflow.set_entry_point("parse_pdf")
    workflow.add_edge("parse_pdf", "detect_doc_type")
    workflow.add_edge("detect_doc_type", "summarize")
    workflow.add_edge("summarize", "persist_metadata")
    workflow.add_edge("persist_metadata", "build_graph")
    workflow.add_edge("build_graph", "index_milvus")
    workflow.add_edge("index_milvus", END)

    return workflow


def build_delete_graph() -> StateGraph:
    """Build the delete workflow subgraph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("delete_paper", delete_paper_node)

    workflow.set_entry_point("delete_paper")
    workflow.add_edge("delete_paper", END)

    return workflow


def build_search_graph() -> StateGraph:
    """Build the search workflow subgraph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("search_papers", search_papers_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.set_entry_point("search_papers")
    workflow.add_edge("search_papers", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow


def build_summarize_graph() -> StateGraph:
    """Build the re-summarize workflow subgraph (for existing papers)."""
    workflow = StateGraph(AgentState)

    workflow.add_node("detect_doc_type", detect_doc_type)
    workflow.add_node("summarize", summarize_paper_node)
    workflow.add_node("persist_metadata", persist_metadata)

    workflow.set_entry_point("detect_doc_type")
    workflow.add_edge("detect_doc_type", "summarize")
    workflow.add_edge("summarize", "persist_metadata")
    workflow.add_edge("persist_metadata", END)

    return workflow


def build_master_graph() -> StateGraph:
    """Build the master graph that routes to sub-workflows based on intent."""
    workflow = StateGraph(AgentState)

    workflow.add_node("route_intent", route_intent)
    workflow.add_node("import_flow", build_import_graph().compile())
    workflow.add_node("delete_flow", build_delete_graph().compile())
    workflow.add_node("search_flow", build_search_graph().compile())
    workflow.add_node("summarize_flow", build_summarize_graph().compile())

    workflow.set_entry_point("route_intent")

    workflow.add_conditional_edges(
        "route_intent",
        route_by_intent,
        {
            "import": "import_flow",
            "delete": "delete_flow",
            "search": "search_flow",
            "summarize": "summarize_flow",
        },
    )

    workflow.add_edge("import_flow", END)
    workflow.add_edge("delete_flow", END)
    workflow.add_edge("search_flow", END)
    workflow.add_edge("summarize_flow", END)

    return workflow


# Compiled graph instance for direct use
literature_agent = build_master_graph().compile()
