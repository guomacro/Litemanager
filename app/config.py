"""Central configuration for the Literature Agent."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PAPER_DIR = DATA_DIR / "papers"
DB_PATH = DATA_DIR / "metadata.db"
MILVUS_DIR = DATA_DIR / "milvus.db"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DIR.mkdir(parents=True, exist_ok=True)

LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
MILVUS_USE_LITE = os.environ.get("MILVUS_USE_LITE", "true").lower() == "true"
