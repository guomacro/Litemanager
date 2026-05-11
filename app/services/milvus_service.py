"""Milvus vector store service for paper chunks and summaries.

Uses Milvus Lite (embedded) by default for MVP, upgradeable to Milvus server.
"""

from typing import Optional

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from openai import OpenAI

from app.config import (
    MILVUS_DIR,
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_USE_LITE,
    EMBED_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

CHUNK_COLLECTION = "paper_chunks"
SUMMARY_COLLECTION = "paper_summaries"

_initialized = False


def _connect() -> None:
    global _initialized
    if _initialized:
        return
    if MILVUS_USE_LITE:
        connections.connect(uri=str(MILVUS_DIR))
    else:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    _initialized = True


def _embed(texts: list[str]) -> list[list[float]]:
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _create_collections() -> None:
    _connect()

    # --- paper_chunks collection ---
    if not utility.has_collection(CHUNK_COLLECTION):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="paper_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="topic_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
        ]
        schema = CollectionSchema(fields, description="Paper text chunks")
        coll = Collection(name=CHUNK_COLLECTION, schema=schema)
        coll.create_index(field_name="embedding", index_params={
            "metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}
        })

    # --- paper_summaries collection ---
    if not utility.has_collection(SUMMARY_COLLECTION):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="paper_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="summary_level", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="summary_text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="topic_path", dtype=DataType.VARCHAR, max_length=512),
        ]
        schema = CollectionSchema(fields, description="Paper summaries")
        coll = Collection(name=SUMMARY_COLLECTION, schema=schema)
        coll.create_index(field_name="embedding", index_params={
            "metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}
        })


def init_milvus() -> None:
    """Initialize Milvus connection and collections."""
    _connect()
    _create_collections()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks on paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para) if current else para
        else:
            if current:
                chunks.append(current[:chunk_size])
            current = para
            while len(current) > chunk_size:
                split_pos = current.rfind(".", 0, chunk_size)
                if split_pos < chunk_size // 4:
                    split_pos = current.rfind(" ", 0, chunk_size)
                if split_pos < chunk_size // 4:
                    split_pos = chunk_size
                chunks.append(current[: split_pos + 1])
                current = current[max(split_pos + 1, split_pos - overlap):]
    if current.strip():
        chunks.append(current[:chunk_size])
    return chunks


def index_paper_chunks(paper_id: str, text: str, topic_path: str = "", doc_type: str = "") -> int:
    """Chunk paper text and insert into Milvus. Returns chunk count."""
    _connect()
    coll = Collection(name=CHUNK_COLLECTION)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = _embed(chunks)
    entities = [
        [f"{paper_id}_chunk_{i}" for i in range(len(chunks))],
        [paper_id] * len(chunks),
        list(range(len(chunks))),
        chunks,
        embeddings,
        [topic_path] * len(chunks),
        [doc_type] * len(chunks),
    ]
    coll.insert(entities)
    coll.flush()
    return len(chunks)


def index_paper_summary(paper_id: str, summary_text: str, level: str = "standard", topic_path: str = "") -> None:
    """Insert a paper summary embedding into Milvus."""
    _connect()
    coll = Collection(name=SUMMARY_COLLECTION)
    chunk_id = f"{paper_id}_summary_{level}"
    embedding = _embed([summary_text])
    entities = [
        [chunk_id],
        [paper_id],
        [level],
        [summary_text[:4096]],
        embedding,
        [topic_path],
    ]
    coll.insert(entities)
    coll.flush()


def search_chunks(query: str, top_k: int = 5, topic_path: str = "", doc_type: str = "") -> list[dict]:
    """Semantic search over paper chunks."""
    _connect()
    coll = Collection(name=CHUNK_COLLECTION)
    coll.load()

    query_embedding = _embed([query])
    expr_parts = []
    if topic_path:
        expr_parts.append(f'topic_path == "{topic_path}"')
    if doc_type:
        expr_parts.append(f'doc_type == "{doc_type}"')
    expr = " && ".join(expr_parts) if expr_parts else None

    results = coll.search(
        data=query_embedding,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        expr=expr,
        output_fields=["paper_id", "chunk_text", "chunk_index", "topic_path", "doc_type"],
    )

    return [
        {
            "id": hit.id,
            "paper_id": hit.entity.get("paper_id", ""),
            "text": hit.entity.get("chunk_text", ""),
            "score": round(hit.score, 4),
            "chunk_index": hit.entity.get("chunk_index", 0),
            "topic_path": hit.entity.get("topic_path", ""),
            "doc_type": hit.entity.get("doc_type", ""),
        }
        for hits in results
        for hit in hits
    ]


def search_summaries(query: str, top_k: int = 5) -> list[dict]:
    """Semantic search over paper summaries."""
    _connect()
    coll = Collection(name=SUMMARY_COLLECTION)
    coll.load()

    query_embedding = _embed([query])
    results = coll.search(
        data=query_embedding,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        output_fields=["paper_id", "summary_text", "summary_level", "topic_path"],
    )

    return [
        {
            "id": hit.id,
            "paper_id": hit.entity.get("paper_id", ""),
            "text": hit.entity.get("summary_text", ""),
            "score": round(hit.score, 4),
            "summary_level": hit.entity.get("summary_level", ""),
            "topic_path": hit.entity.get("topic_path", ""),
        }
        for hits in results
        for hit in hits
    ]


def delete_paper_vectors(paper_id: str) -> None:
    """Remove all vectors for a paper from both collections."""
    _connect()
    for name in [CHUNK_COLLECTION, SUMMARY_COLLECTION]:
        if not utility.has_collection(name):
            continue
        coll = Collection(name=name)
        coll.load()
        coll.delete(f'paper_id == "{paper_id}"')
