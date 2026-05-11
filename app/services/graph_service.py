"""Graph service: build and maintain paper-topic-concept relationships."""

from app.services.metadata_service import (
    upsert_graph_node,
    insert_graph_edge,
)


def build_paper_graph(paper_id: str, title: str, topic_path: str, doc_type: str) -> None:
    """Create graph nodes and edges for a paper.

    Creates:
    - A 'paper' node for the paper itself
    - 'topic' nodes along the topic_path hierarchy
    - 'doc_type' node
    - Edges: paper -> topic (belongs_to), topic -> parent_topic (subtopic_of), paper -> doc_type (has_type)
    """
    # Paper node
    upsert_graph_node(paper_id, "paper", title, {"doc_type": doc_type})

    # Topic hierarchy nodes
    parts = [p.strip() for p in topic_path.split("/") if p.strip()]
    cumulative: list[str] = []
    prev_id = None
    for part in parts:
        cumulative.append(part)
        node_id = "/".join(cumulative)
        upsert_graph_node(node_id, "topic", part)
        if prev_id:
            insert_graph_edge(prev_id, node_id, "subtopic_of")
        prev_id = node_id

    # Link paper to deepest topic
    if prev_id:
        insert_graph_edge(paper_id, prev_id, "belongs_to")

    # Link paper to doc_type
    type_node_id = f"type:{doc_type}"
    upsert_graph_node(type_node_id, "doc_type", doc_type)
    insert_graph_edge(paper_id, type_node_id, "has_type")
