"""Gradio-based GUI for the Literature Management Agent (LangGraph + Milvus)."""

import os
import tempfile
from pathlib import Path

import gradio as gr  # type: ignore

from app.state import AgentState
from app.graph import literature_agent
from app.services.metadata_service import (
    init_metadata_db,
    list_papers,
    get_paper,
    get_summary,
    get_paper_graph_neighbors,
)
from app.services.milvus_service import init_milvus


def _init() -> None:
    init_metadata_db()
    init_milvus()


def _run_agent(state: dict) -> dict:
    """Run the LangGraph agent with the given state and return the final state."""
    result = literature_agent.invoke(state)
    return result


def on_import(file_obj, summary_level: str) -> str:
    """Handle PDF import via LangGraph import workflow."""
    if file_obj is None:
        return "**Error**: Please upload a PDF file."

    temp_path = None
    try:
        suffix = Path(file_obj.name).suffix if hasattr(file_obj, "name") else ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = file_obj.read() if hasattr(file_obj, "read") else file_obj
            tmp.write(content)
            temp_path = tmp.name

        state: AgentState = {
            "intent": "import",
            "pdf_path": temp_path,
            "summary_level": summary_level,
            "parsed_metadata": None,
            "doc_type": "",
            "topic_path": "",
            "summary_data": None,
            "query": "",
            "topic_filter": "",
            "type_filter": "",
            "top_k": 5,
            "search_results": [],
            "paper_id": "",
            "hard_delete": False,
            "result": {},
            "error": "",
            "messages": [],
        }

        final_state = _run_agent(state)

        if final_state.get("error"):
            return f"**Import Failed**: {final_state['error']}"

        result = final_state.get("result", {})
        summary = result.get("summary", {})

        lines = [
            f"## Import Success",
            f"",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Paper ID | `{result.get('paper_id', '?')}` |",
            f"| Title | {result.get('title', '?')} |",
            f"| Type | {result.get('doc_type', '?')} |",
            f"| Topic Path | `{result.get('topic_path', '?')}` |",
        ]

        chunks = result.get("chunks_indexed")
        if chunks is not None:
            lines.append(f"| Chunks Indexed | {chunks} |")

        lines.append(f"")
        lines.append(f"### Summary")
        lines.append(f"{summary.get('paper_summary', 'N/A')}")

        key_points = summary.get("key_points", [])
        if key_points:
            lines.append(f"\n### Key Points")
            for pt in key_points:
                lines.append(f"- {pt}")

        return "\n".join(lines)

    except Exception as e:
        return f"**Import Failed**: {str(e)}"
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def on_search(query: str, topic_filter: str, type_filter: str, top_k: int) -> str:
    """Handle search via LangGraph search workflow."""
    if not query.strip():
        return "Please enter a search query."

    try:
        state: AgentState = {
            "intent": "search",
            "query": query,
            "topic_filter": topic_filter.strip(),
            "type_filter": type_filter.strip(),
            "top_k": top_k,
            "pdf_path": "",
            "summary_level": "standard",
            "parsed_metadata": None,
            "doc_type": "",
            "topic_path": "",
            "summary_data": None,
            "search_results": [],
            "paper_id": "",
            "hard_delete": False,
            "result": {},
            "error": "",
            "messages": [],
        }

        final_state = _run_agent(state)

        if final_state.get("error"):
            return f"**Search Failed**: {final_state['error']}"

        result = final_state.get("result", {})
        answer = result.get("answer", "")
        results = result.get("results", [])

        lines = [f"## {answer}", f""]

        if results:
            lines.append(f"### Matched Papers ({len(results)} found)")
            for i, r in enumerate(results):
                lines.append(f"**{i+1}. {r.get('title', 'Unknown')}**")
                lines.append(f"- Paper ID: `{r.get('paper_id', '')}` | Score: {r.get('score', 0):.4f}")
                lines.append(f"- Type: {r.get('doc_type', '')} | Topic: `{r.get('topic_path', '')}`")
                lines.append(f"- Match type: {r.get('match_type', '')}")
                lines.append(f"> {r.get('match_text', '')[:300]}...")
                lines.append("")
        else:
            lines.append("No matching papers found.")

        return "\n".join(lines)

    except Exception as e:
        return f"**Search Failed**: {str(e)}"


def on_list_papers(search_term: str) -> str:
    """List all papers in the library."""
    try:
        papers = list_papers(search=search_term.strip() or None, limit=100)
        if not papers:
            return "No papers found. Upload a PDF to get started."

        lines = [
            "| Paper ID | Title | Type | Topic Path |",
            "|----------|-------|------|------------|",
        ]
        for p in papers:
            title = p["title"][:60]
            topic = p["topic_path"][:40] if p["topic_path"] else "-"
            lines.append(f"| {p['paper_id']} | {title} | {p['doc_type']} | {topic} |")

        return "\n".join(lines)

    except Exception as e:
        return f"**Error**: {str(e)}"


def on_view_paper(paper_id: str) -> str:
    """View full paper details including summary and graph neighbors."""
    if not paper_id.strip():
        return "Please enter a Paper ID."

    try:
        detail = get_paper(paper_id.strip())
        if not detail:
            return f"Paper `{paper_id.strip()}` not found."

        summary = get_summary(paper_id.strip(), "standard") or get_summary(paper_id.strip(), "quick")
        neighbors = get_paper_graph_neighbors(paper_id.strip())

        lines = [
            f"## {detail.get('title', 'Unknown')}",
            f"",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Paper ID | `{detail.get('paper_id', '')}` |",
            f"| Type | {detail.get('doc_type', '')} |",
            f"| Topic Path | `{detail.get('topic_path', '')}` |",
            f"| Keywords | {detail.get('keywords') or '-'} |",
            f"| File | {detail.get('file_name') or '-'} |",
            f"| Created | {detail.get('created_at', '')} |",
            f"",
            f"### Abstract",
            f"",
            detail.get("abstract", "No abstract available.") or "No abstract available.",
        ]

        if summary:
            lines.append(f"\n### Summary")
            lines.append(str(summary.get("paper_summary", "")))
            key_points = summary.get("key_points", [])
            if key_points:
                lines.append("\n### Key Points")
                for pt in key_points:
                    lines.append(f"- {pt}")
            section_summaries = summary.get("section_summaries", [])
            if section_summaries:
                lines.append("\n### Section Summaries")
                for s in section_summaries:
                    lines.append(f"- **{s.get('section', '')}**: {s.get('summary', '')}")

        if neighbors:
            lines.append("\n### Graph Relations")
            for n in neighbors[:20]:
                lines.append(f"- `{n['edge_type']}`: {n.get('source_label', '')} -> {n.get('target_label', '')}")

        return "\n".join(lines)

    except Exception as e:
        return f"**Error**: {str(e)}"


def on_delete(paper_id: str, hard_delete: bool) -> str:
    """Delete a paper via LangGraph delete workflow."""
    if not paper_id.strip():
        return "Please enter a Paper ID."

    try:
        state: AgentState = {
            "intent": "delete",
            "paper_id": paper_id.strip(),
            "hard_delete": hard_delete,
            "pdf_path": "",
            "summary_level": "standard",
            "parsed_metadata": None,
            "doc_type": "",
            "topic_path": "",
            "summary_data": None,
            "query": "",
            "topic_filter": "",
            "type_filter": "",
            "top_k": 5,
            "search_results": [],
            "result": {},
            "error": "",
            "messages": [],
        }

        final_state = _run_agent(state)

        if final_state.get("error"):
            return f"**Delete Failed**: {final_state['error']}"

        result = final_state.get("result", {})
        status = result.get("status", "unknown")
        title = result.get("title", paper_id)

        if status == "not_found":
            return f"Paper `{paper_id.strip()}` not found."
        return f"Paper **{title}** (`{paper_id.strip()}`) **{status}**."

    except Exception as e:
        return f"**Delete Failed**: {str(e)}"


def create_ui() -> gr.Blocks:
    """Build the Gradio interface with 5 tabs."""
    css = """
    .paper-table table { width: 100%; }
    .paper-table th { text-align: left; background: #f5f5f5; }
    """

    with gr.Blocks(title="LiteManager - Literature Agent", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown(
            """
            # LiteManager - Literature Management Agent
            **LangGraph** + **Milvus** + **SQLite** | Upload, summarize, organize, and search your research papers.
            """
        )

        with gr.Tabs():
            # ---- Import Tab ----
            with gr.TabItem("Import PDF"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="binary")
                        summary_level = gr.Dropdown(
                            choices=["quick", "standard", "deep"],
                            value="standard",
                            label="Summary Level",
                        )
                        import_btn = gr.Button("Import & Analyze", variant="primary")
                    with gr.Column(scale=2):
                        import_output = gr.Markdown(
                            value="Upload a PDF to import it into the library.",
                            label="Result",
                        )
                import_btn.click(fn=on_import, inputs=[file_input, summary_level], outputs=import_output)

            # ---- Search Tab ----
            with gr.TabItem("Search"):
                with gr.Row():
                    with gr.Column(scale=1):
                        search_query = gr.Textbox(
                            label="Search Query",
                            placeholder="e.g., retrieval augmented generation for knowledge-intensive tasks",
                        )
                        topic_filter = gr.Textbox(label="Filter by Topic Path (optional)", placeholder="e.g., AI/NLP")
                        type_filter = gr.Dropdown(
                            choices=["", "survey", "method", "application", "benchmark", "theory"],
                            value="",
                            label="Filter by Paper Type",
                        )
                        top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Max Results")
                        search_btn = gr.Button("Search", variant="primary")
                    with gr.Column(scale=2):
                        search_output = gr.Markdown(label="Search Results")
                search_btn.click(
                    fn=on_search,
                    inputs=[search_query, topic_filter, type_filter, top_k],
                    outputs=search_output,
                )

            # ---- Library Tab ----
            with gr.TabItem("Library"):
                search_term = gr.Textbox(label="Search by Title/Keywords", placeholder="Optional filter")
                refresh_btn = gr.Button("Refresh", variant="secondary")
                library_output = gr.Markdown(
                    value="Click **Refresh** to load papers.",
                    elem_classes=["paper-table"],
                )
                refresh_btn.click(fn=on_list_papers, inputs=search_term, outputs=library_output)

            # ---- View Details Tab ----
            with gr.TabItem("View Details"):
                view_paper_id = gr.Textbox(label="Paper ID", placeholder="Enter paper ID to view details")
                view_btn = gr.Button("View Details", variant="primary")
                detail_output = gr.Markdown(label="Paper Details")
                view_btn.click(fn=on_view_paper, inputs=view_paper_id, outputs=detail_output)

            # ---- Delete Tab ----
            with gr.TabItem("Delete"):
                delete_paper_id = gr.Textbox(label="Paper ID", placeholder="Enter paper ID to delete")
                hard_delete_cb = gr.Checkbox(label="Hard Delete (permanent, removes all data)", value=False)
                delete_btn = gr.Button("Delete", variant="stop")
                delete_output = gr.Markdown(label="Delete Result")
                delete_btn.click(
                    fn=on_delete,
                    inputs=[delete_paper_id, hard_delete_cb],
                    outputs=delete_output,
                )

        _init()

    return demo


if __name__ == "__main__":
    _init()
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
