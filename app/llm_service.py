"""LLM service: paper classification, summarization, topic path generation, query rewrite."""

import json
import re
from pathlib import Path

from openai import OpenAI

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, PROMPTS_DIR


def _load_prompt(name: str) -> str:
    prompt_path = PROMPTS_DIR / name
    if prompt_path.exists():
        return prompt_path.read_text()
    return ""


def _get_client() -> OpenAI:
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def _chat(system: str, prompt: str, temperature: float = 0.3) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def classify_doc_type(title: str, abstract: str, keywords: str) -> str:
    """Classify paper as survey / method / application / benchmark / theory."""
    system = """Classify academic papers into exactly one type:
- survey: comprehensive review of a field
- method: proposes a new method, model, or algorithm
- application: applies existing techniques to a specific problem
- benchmark: introduces datasets, benchmarks, evaluation frameworks
- theory: theoretical analysis, proofs, formal frameworks

Return ONLY the type name, nothing else."""

    prompt = f"Title: {title}\n\nAbstract: {abstract[:1500]}\n\nKeywords: {keywords}"
    result = _chat(system, prompt, temperature=0.1).strip().lower()
    valid = {"survey", "method", "application", "benchmark", "theory"}
    for v in valid:
        if v in result:
            return v
    return "method"


def generate_topic_path(title: str, abstract: str, keywords: str) -> str:
    """Generate hierarchical topic path like 'AI/NLP/Question-Answering/RAG'."""
    system = """You are a research librarian. Given paper metadata, output its topic path:
Area/Subfield/Topic/Subtopic

Use established CS/AI taxonomy, 3-4 levels. Output only the path.
Examples: AI/NLP/Question-Answering/RAG, AI/CV/Object-Detection, Systems/Databases/Query-Optimization"""

    prompt = f"Title: {title}\n\nAbstract: {abstract[:1500]}\n\nKeywords: {keywords}"
    return _chat(system, prompt, temperature=0.2).strip().strip("'\"")


def summarize_paper(
    title: str, abstract: str, sections: list[dict],
    doc_type: str, level: str = "standard",
) -> dict:
    """Generate structured summary based on paper type and detail level.

    Returns dict with: paper_summary, key_points, section_summaries
    """
    template = _load_prompt("summarize.md") or _default_summary_template()

    sections_text = "\n\n".join(
        f"## {s['heading']}\n{s['content'][:800]}" for s in sections[:10]
    )

    prompt = (
        template.format(title=title, abstract=abstract, doc_type=doc_type, level=level)
        + f"\n\nPaper Sections:\n{sections_text}"
    )

    system = f"You are an expert academic summarizer. Generate a {level}-level summary for a {doc_type} paper."
    response = _chat(system, prompt, temperature=0.3)
    return _parse_json_response(response)


def rewrite_query(raw_query: str) -> str:
    """Rewrite a user query for better retrieval performance."""
    system = """Rewrite the user's search query to be more precise and keyword-rich for vector retrieval.
Expand abbreviations, add synonyms, and structure as a concise search phrase.
Return ONLY the rewritten query, nothing else."""

    return _chat(system, raw_query, temperature=0.2).strip().strip("'\"")


def generate_search_answer(query: str, results: list[dict]) -> str:
    """Generate a synthesized answer from search results."""
    if not results:
        return "No relevant papers found."

    results_text = "\n\n---\n\n".join(
        f"Paper: {r.get('title', 'Unknown')}\n"
        f"Type: {r.get('doc_type', '?')} | Topic: {r.get('topic_path', '?')}\n"
        f"Relevance Score: {r.get('score', 0):.4f}\n"
        f"Snippet: {r.get('match_text', r.get('text', ''))[:600]}"
        for r in results[:5]
    )

    system = "You are a helpful research assistant. Synthesize search results into a concise answer."
    prompt = (
        f"User Query: {query}\n\n"
        f"Search Results:\n{results_text}\n\n"
        "Provide: (1) a 2-3 sentence synthesis answering the query, "
        "(2) list the most relevant papers with a 1-line explanation of why each matches."
    )
    return _chat(system, prompt, temperature=0.4)


def _default_summary_template() -> str:
    return """Summarize the following {doc_type} paper at {level} detail level.

Title: {title}
Abstract: {abstract}

Return JSON:
{{
  "paper_summary": "200-400 word comprehensive summary",
  "key_points": ["5 key takeaways"],
  "section_summaries": [{{"section": "Name", "summary": "1-2 sentences"}}]
}}
Return ONLY the JSON, no other text."""


def _parse_json_response(response: str) -> dict:
    response = response.strip()
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return {"paper_summary": response[:800], "key_points": [], "section_summaries": []}
