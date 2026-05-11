# LiteManager - Literature Management Agent

A minimal viable literature management system built with **LangGraph**, **Milvus**, and **SQLite**. Features AI-powered PDF parsing, multi-level summarization, topic-graph organization, and semantic search.

## Architecture

```
LiteManager/
├── app/
│   ├── config.py              # Central configuration (paths, models, API keys)
│   ├── state.py               # LangGraph AgentState definition
│   ├── graph.py               # LangGraph workflow orchestration
│   ├── llm_service.py         # LLM calls (classification, summary, topic path, query rewrite)
│   ├── nodes/
│   │   ├── route_intent.py    # Intent routing node
│   │   ├── parse_pdf.py       # PDF text/metadata extraction
│   │   ├── summarize.py       # Doc type detection + structured summarization
│   │   ├── build_graph.py     # Build topic/doc_type graph relationships
│   │   ├── index_milvus.py    # Chunk + embed + insert into Milvus
│   │   ├── persist_metadata.py # Save metadata + summary to SQLite
│   │   ├── delete_paper.py    # Cascading delete (vectors, graph, metadata, file)
│   │   └── search_papers.py   # Query rewrite + summary-first search + answer generation
│   └── services/
│       ├── pdf_service.py     # PyMuPDF PDF parsing
│       ├── milvus_service.py  # Milvus vector store (chunks + summaries collections)
│       ├── metadata_service.py # SQLite metadata + graph CRUD
│       └── graph_service.py   # Build paper-topic-concept graph edges
├── gui/
│   └── app.py                 # Gradio web interface (5 tabs)
├── prompts/
│   ├── summarize.md           # Summarization prompt template
│   └── classify.md            # Classification prompt template
├── data/                      # Runtime data (SQLite DB, Milvus, PDF copies)
├── requirements.txt
└── README.md
```

## LangGraph Workflows

```
                    ┌─────────────┐
                    │ route_intent│
                    └──────┬──────┘
           ┌───────────────┼───────────────┬──────────────┐
           ▼               ▼               ▼              ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐   ┌──────────┐
    │  import  │    │  delete  │    │  search  │   │summarize │
    └────┬─────┘    └────┬─────┘    └────┬─────┘   └────┬─────┘
         │               │               │              │
    parse_pdf        delete_paper    search_papers   detect_doc_type
         │               │               │              │
    detect_doc_type      END        generate_answer  summarize
         │                               │              │
    summarize                           END       persist_metadata
         │                                             │
    persist_metadata                                   END
         │
    build_graph
         │
    index_milvus
         │
         END
```

### Import Workflow
1. `parse_pdf` - Extract text, title, abstract, keywords, sections via PyMuPDF
2. `detect_doc_type` - Classify as survey/method/application/benchmark/theory via LLM; generate topic_path
3. `summarize` - Generate structured summary (paper_summary, key_points, section_summaries)
4. `persist_metadata` - Write paper record + summary to SQLite, copy PDF to data/
5. `build_graph` - Create graph nodes (paper, topic hierarchy, doc_type) and edges
6. `index_milvus` - Chunk text, embed, insert into Milvus `paper_chunks` and `paper_summaries` collections

### Search Workflow
1. `search_papers` - Rewrite query via LLM, search summaries first, then chunks; deduplicate and rank
2. `generate_answer` - Synthesize results into a concise answer via LLM

### Delete Workflow
1. `delete_paper` - Locate paper; soft delete (mark deleted) or hard delete (remove vectors, graph, file, metadata)

## Data Storage

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Metadata | SQLite | Paper records, graph nodes/edges, summaries |
| Vectors | Milvus | `paper_chunks` collection (text chunks + embeddings) and `paper_summaries` collection |
| Files | Local disk | PDF files stored under `data/papers/` |

### SQLite Tables
- **papers**: paper_id, title, authors, abstract, keywords, source, doc_type, topic_path, file_path, status, timestamps
- **paper_summaries**: paper_id, summary_level, summary_data (JSON), created_at
- **graph_nodes**: node_id, node_type (paper/topic/doc_type), label, props
- **graph_edges**: edge_id, source_id, target_id, edge_type (belongs_to/subtopic_of/has_type), props

### Milvus Collections
- **paper_chunks**: chunk_id, paper_id, chunk_index, chunk_text, embedding (1536d), topic_path, doc_type
- **paper_summaries**: summary_id, paper_id, summary_level, summary_text, embedding (1536d), topic_path

## Setup

```bash
conda create -n agent3.12 python=3.12
conda activate agent3.12
pip install -r requirements.txt
```

Configure your OpenAI-compatible API credentials:
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"
export EMBED_MODEL="text-embedding-3-small"
```

Milvus runs in **Lite** mode by default (embedded, no server required). For production, set:
```bash
export MILVUS_USE_LITE=false
export MILVUS_HOST=localhost
export MILVUS_PORT=19530
```

## Usage

```bash
python -m gui.app
```

Open http://127.0.0.1:7860 in your browser.

### GUI Tabs
| Tab | Function |
|-----|----------|
| Import PDF | Upload a PDF, select summary level (quick/standard/deep), trigger full import pipeline |
| Search | Semantic search with topic/type filters, AI-synthesized answers |
| Library | Browse all papers with optional keyword filter |
| View Details | Inspect paper metadata, summary, and graph relationships |
| Delete | Soft delete (recoverable) or hard delete (permanent cascading removal) |

## MVP Features
- [x] PDF import with auto-extracted metadata (title, abstract, keywords, sections)
- [x] AI document type classification (survey/method/application/benchmark/theory)
- [x] Automatic topic path generation (e.g. `AI/NLP/Question-Answering/RAG`)
- [x] Multi-level structured summarization (quick/standard/deep)
- [x] Graph-based topic organization (paper -> topic -> subtopic hierarchy)
- [x] Milvus vector indexing (chunks + summaries, cosine similarity)
- [x] Semantic search with query rewriting and AI-synthesized answers
- [x] Soft/hard delete with cascading cleanup (vectors, graph, metadata, file)
- [x] Gradio web GUI

## Roadmap
1. ✅ **Phase 1 (MVP)**: Import, parse, summarize, search, delete
2. **Phase 2 (Graph RAG)**: Enhanced topic-graph traversal + vector hybrid retrieval
3. **Phase 3 (Lifecycle)**: Batch operations, export, version history
4. **Phase 4 (External)**: Arxiv API integration, candidate paper import
5. **Phase 5 (Advanced)**: OCR support, Neo4j migration, multi-user
