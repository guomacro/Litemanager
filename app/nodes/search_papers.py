"""LangGraph nodes: search papers using graph-first, vector-fallback strategy."""

from app.state import AgentState
from app.llm_service import rewrite_query, generate_search_answer
from app.services.milvus_service import (
    init_milvus,
    search_summaries,
    search_chunks,
)
from app.services.metadata_service import get_paper


def search_papers_node(state: AgentState) -> AgentState:
    """Execute search: rewrite query, search summaries then chunks, deduplicate and rank."""
    query = state.get("query", "")
    topic_filter = state.get("topic_filter", "")
    type_filter = state.get("type_filter", "")
    top_k = state.get("top_k", 5)

    if not query.strip():
        state["error"] = "No search query provided."
        state["search_results"] = []
        return state

    try:
        init_milvus()

        # Rewrite query for better retrieval
        rewritten = rewrite_query(query)

        results: list[dict] = []
        seen: set[str] = set()

        # Search summaries first (cheaper, good for broad matching)
        for sr in search_summaries(rewritten, top_k=top_k):
            pid = sr["paper_id"]
            paper = get_paper(pid)
            if paper and pid not in seen:
                seen.add(pid)
                results.append({
                    "paper_id": pid,
                    "title": paper["title"],
                    "doc_type": paper["doc_type"],
                    "topic_path": paper["topic_path"],
                    "match_type": "summary",
                    "match_text": sr["text"][:500],
                    "score": sr["score"],
                })

        # Search chunks for fine-grained matches
        for cr in search_chunks(rewritten, top_k=top_k, topic_path=topic_filter, doc_type=type_filter):
            pid = cr["paper_id"]
            if pid not in seen:
                seen.add(pid)
                paper = get_paper(pid)
                if paper:
                    results.append({
                        "paper_id": pid,
                        "title": paper["title"],
                        "doc_type": paper["doc_type"],
                        "topic_path": paper["topic_path"],
                        "match_type": "chunk",
                        "match_text": cr["text"][:500],
                        "score": cr["score"],
                    })

        results.sort(key=lambda r: r["score"], reverse=True)
        state["search_results"] = results[:top_k]
        state["error"] = ""
    except Exception as e:
        state["error"] = f"Search failed: {str(e)}"
        state["search_results"] = []

    return state


def generate_answer_node(state: AgentState) -> AgentState:
    """Generate a synthesized answer from search results."""
    query = state.get("query", "")
    results = state.get("search_results", [])

    try:
        answer = generate_search_answer(query, results)
        state["result"] = {
            "query": query,
            "answer": answer,
            "results": results,
            "total": len(results),
        }
    except Exception as e:
        state["result"] = {
            "query": query,
            "answer": f"Search completed but answer generation failed: {e}",
            "results": results,
            "total": len(results),
        }

    return state
