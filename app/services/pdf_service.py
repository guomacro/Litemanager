"""PDF parsing service using PyMuPDF."""

import re
from pathlib import Path
from typing import Optional


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract all text from a PDF file page by page."""
    import fitz

    doc = fitz.open(file_path)
    full_text: list[str] = []
    for page in doc:
        text = page.get_text("text")
        if text:
            full_text.append(text)
    doc.close()
    return "\n".join(full_text)


def extract_metadata_from_text(text: str) -> dict:
    """Heuristically extract title, abstract, keywords, and sections from paper text."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    title = _extract_title(lines)
    abstract = _extract_abstract(text)
    keywords = _extract_keywords(text)
    sections = _extract_sections(text)

    return {
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "sections": sections,
        "full_text": text,
    }


def _extract_title(lines: list[str]) -> str:
    for line in lines[:20]:
        clean = line.strip()
        if len(clean) > 5 and not clean.lower().startswith(
            ("abstract", "keywords", "introduction", "figure", "table")
        ):
            return clean
    return "Unknown Title"


def _extract_abstract(text: str) -> str:
    patterns = [
        r"(?i)abstract\s*[\n\r]+(.+?)(?:\n\s*(?:\d+[\.\s]+)?(?:introduction|related work|background|1\.))",
        r"(?i)abstract\s*[\n\r]+(.+?)(?:\n\n)",
        r"(?i)abstract[：:\s\-]+(.+?)(?:\n\s*\n)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            if len(abstract) > 20:
                return abstract[:3000]
    return ""


def _extract_keywords(text: str) -> str:
    patterns = [
        r"(?i)keywords[：:\s\-]+(.+?)(?:\n)",
        r"(?i)key words[：:\s\-]+(.+?)(?:\n)",
        r"(?i)index terms[：:\s\-]+(.+?)(?:\n)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def _extract_sections(text: str) -> list[dict]:
    section_pattern = re.compile(r"(?m)^\s*(?:\d+\.?\s+)?([A-Z][A-Za-z\s\-]{2,50})$")
    matches = list(section_pattern.finditer(text))
    sections: list[dict] = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if len(heading) < 60 and len(content) > 50:
            sections.append({"heading": heading, "content": content[:2000]})
    return sections
