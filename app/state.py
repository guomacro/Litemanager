"""LangGraph state definition for the Literature Agent."""

from typing import TypedDict, Annotated, Optional, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State shared across all LangGraph nodes.

    Fields are populated progressively as the workflow executes.
    """

    # --- Routing ---
    intent: str  # import | delete | search | summarize

    # --- Import fields ---
    pdf_path: str
    parsed_metadata: Optional[dict]  # {title, abstract, keywords, sections, full_text}
    doc_type: str  # survey | method | application | benchmark | theory
    topic_path: str
    summary_level: str  # quick | standard | deep
    summary_data: Optional[dict]  # {paper_summary, key_points, section_summaries}

    # --- Search fields ---
    query: str
    topic_filter: str
    type_filter: str
    top_k: int
    search_results: list[dict]

    # --- Delete fields ---
    paper_id: str
    hard_delete: bool

    # --- Result ---
    result: dict  # Final output returned to caller
    error: str

    # --- Messages (LangGraph standard) ---
    messages: Annotated[Sequence[BaseMessage], add_messages]
