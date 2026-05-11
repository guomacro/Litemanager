"""SQLite metadata service for paper records and graph relationships."""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import sqlite3 as _sqlite3

from app.config import DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _conn() -> _sqlite3.Connection:
    c = _sqlite3.connect(str(DB_PATH))
    c.row_factory = _sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c


def init_metadata_db() -> None:
    """Create all tables if they don't exist."""
    c = _conn()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id    TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            authors     TEXT DEFAULT '',
            abstract    TEXT DEFAULT '',
            keywords    TEXT DEFAULT '',
            source      TEXT DEFAULT 'upload',
            source_id   TEXT DEFAULT '',
            doc_type    TEXT DEFAULT 'method',
            topic_path  TEXT DEFAULT '',
            file_path   TEXT DEFAULT '',
            file_name   TEXT DEFAULT '',
            status      TEXT DEFAULT 'active',
            created_at  TEXT,
            updated_at  TEXT,
            deleted_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS paper_summaries (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id      TEXT NOT NULL,
            summary_level TEXT NOT NULL,
            summary_data  TEXT NOT NULL,
            created_at    TEXT,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS graph_nodes (
            node_id   TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            label     TEXT NOT NULL,
            props     TEXT DEFAULT '{}',
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS graph_edges (
            edge_id    TEXT PRIMARY KEY,
            source_id  TEXT NOT NULL,
            target_id  TEXT NOT NULL,
            edge_type  TEXT NOT NULL,
            props      TEXT DEFAULT '{}',
            FOREIGN KEY (source_id) REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES graph_nodes(node_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
        CREATE INDEX IF NOT EXISTS idx_papers_doctype ON papers(doc_type);
        CREATE INDEX IF NOT EXISTS idx_papers_topic ON papers(topic_path);
        CREATE INDEX IF NOT EXISTS idx_summaries_paper ON paper_summaries(paper_id);
        CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_id);
    """)
    c.commit()
    c.close()


# --- Paper CRUD ---

def insert_paper(
    title: str,
    abstract: str = "",
    keywords: str = "",
    doc_type: str = "method",
    topic_path: str = "",
    file_path: str = "",
    file_name: str = "",
    source: str = "upload",
    source_id: str = "",
    authors: str = "",
    paper_id: Optional[str] = None,
) -> str:
    paper_id = paper_id or _uid()
    now = _now()
    c = _conn()
    c.execute(
        """INSERT INTO papers (paper_id, title, authors, abstract, keywords,
           source, source_id, doc_type, topic_path, file_path, file_name,
           status, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (paper_id, title, authors, abstract, keywords, source, source_id,
         doc_type, topic_path, file_path, file_name, "active", now, now),
    )
    c.commit()
    c.close()
    return paper_id


def get_paper(paper_id: str) -> Optional[dict]:
    c = _conn()
    row = c.execute(
        "SELECT * FROM papers WHERE paper_id=? AND status='active'", (paper_id,)
    ).fetchone()
    c.close()
    return dict(row) if row else None


def list_papers(
    doc_type: Optional[str] = None,
    topic_path: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    c = _conn()
    query = "SELECT * FROM papers WHERE status='active'"
    params: list = []

    if doc_type:
        query += " AND doc_type=?"
        params.append(doc_type)
    if topic_path:
        query += " AND topic_path LIKE ?"
        params.append(f"%{topic_path}%")
    if search:
        query += " AND (title LIKE ? OR abstract LIKE ? OR keywords LIKE ?)"
        params.extend([f"%{search}%" for _ in range(3)])

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    rows = c.execute(query, params).fetchall()
    c.close()
    return [dict(r) for r in rows]


def soft_delete_paper(paper_id: str) -> bool:
    now = _now()
    c = _conn()
    cur = c.execute(
        "UPDATE papers SET status='deleted', deleted_at=?, updated_at=? WHERE paper_id=? AND status='active'",
        (now, now, paper_id),
    )
    c.commit()
    affected = cur.rowcount
    c.close()
    return affected > 0


def hard_delete_paper(paper_id: str) -> bool:
    c = _conn()
    cur = c.execute("DELETE FROM papers WHERE paper_id=?", (paper_id,))
    c.commit()
    affected = cur.rowcount
    c.close()
    return affected > 0


# --- Summary ---

def save_summary(paper_id: str, level: str, summary_data: dict) -> None:
    c = _conn()
    c.execute(
        "INSERT INTO paper_summaries (paper_id, summary_level, summary_data, created_at) VALUES (?,?,?,?)",
        (paper_id, level, json.dumps(summary_data, ensure_ascii=False), _now()),
    )
    c.commit()
    c.close()


def get_summary(paper_id: str, level: str = "standard") -> Optional[dict]:
    c = _conn()
    row = c.execute(
        "SELECT summary_data FROM paper_summaries WHERE paper_id=? AND summary_level=? ORDER BY created_at DESC LIMIT 1",
        (paper_id, level),
    ).fetchone()
    c.close()
    return json.loads(row["summary_data"]) if row else None


# --- Graph ---

def upsert_graph_node(node_id: str, node_type: str, label: str, props: Optional[dict] = None) -> None:
    c = _conn()
    c.execute(
        """INSERT INTO graph_nodes (node_id, node_type, label, props, created_at)
           VALUES (?,?,?,?,?) ON CONFLICT(node_id) DO UPDATE
           SET label=excluded.label, props=excluded.props""",
        (node_id, node_type, label, json.dumps(props or {}, ensure_ascii=False), _now()),
    )
    c.commit()
    c.close()


def insert_graph_edge(source_id: str, target_id: str, edge_type: str, props: Optional[dict] = None) -> str:
    edge_id = _uid()
    c = _conn()
    c.execute(
        "INSERT OR IGNORE INTO graph_edges (edge_id, source_id, target_id, edge_type, props) VALUES (?,?,?,?,?)",
        (edge_id, source_id, target_id, edge_type, json.dumps(props or {}, ensure_ascii=False)),
    )
    c.commit()
    c.close()
    return edge_id


def delete_graph_for_paper(paper_id: str) -> None:
    c = _conn()
    c.execute("DELETE FROM graph_edges WHERE source_id=? OR target_id=?", (paper_id, paper_id))
    c.execute("DELETE FROM graph_nodes WHERE node_id=?", (paper_id,))
    c.commit()
    c.close()


def get_paper_graph_neighbors(paper_id: str) -> list[dict]:
    c = _conn()
    rows = c.execute(
        """SELECT e.edge_type, e.source_id, e.target_id,
                  ns.label as source_label, nt.label as target_label
           FROM graph_edges e
           JOIN graph_nodes ns ON ns.node_id = e.source_id
           JOIN graph_nodes nt ON nt.node_id = e.target_id
           WHERE e.source_id=? OR e.target_id=?""",
        (paper_id, paper_id),
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]
