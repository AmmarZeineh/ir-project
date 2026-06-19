"""
SQLite-backed raw document store.

The corpora are pre-processed and indexed as CSV/pickle files for building
the indexes, but the *raw* (original, unprocessed) document text that is
shown to the user and used by RAG is stored in a small SQLite database per
dataset and read by `doc_id` at query time. This keeps result/RAG lookups
fast and avoids loading the full corpus CSV into memory.
"""

import csv
import sqlite3
from pathlib import Path

DOCSTORE_DIR = Path("data/docstore")


def docstore_path(dataset: str) -> Path:
    DOCSTORE_DIR.mkdir(parents=True, exist_ok=True)
    return DOCSTORE_DIR / f"{dataset}.db"


def build_docstore(dataset: str, docs_csv_path: Path | str, chunk_size: int = 5000) -> int:
    """
    Loads a `docs.csv` / `docs_relevant.csv` file (columns: doc_id, text)
    into a SQLite database for the given dataset.

    Returns the number of rows inserted.
    """
    db_path = docstore_path(dataset)
    db_path.unlink(missing_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE docs (doc_id TEXT PRIMARY KEY, text TEXT NOT NULL)"
    )

    total = 0
    with open(docs_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            batch.append((str(row["doc_id"]), str(row.get("text", ""))))
            if len(batch) >= chunk_size:
                conn.executemany(
                    "INSERT OR REPLACE INTO docs (doc_id, text) VALUES (?, ?)",
                    batch,
                )
                total += len(batch)
                batch = []
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO docs (doc_id, text) VALUES (?, ?)",
                batch,
            )
            total += len(batch)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON docs (doc_id)")
    conn.commit()
    conn.close()
    return total


class DocStore:
    """Read-only handle for fetching raw document text by doc_id."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        self._conn = sqlite3.connect(docstore_path(dataset), check_same_thread=False)

    def get(self, doc_id: str, default: str = "") -> str:
        row = self._conn.execute(
            "SELECT text FROM docs WHERE doc_id = ?", (str(doc_id),)
        ).fetchone()
        return row[0] if row else default

    def get_many(self, doc_ids: list[str]) -> dict[str, str]:
        if not doc_ids:
            return {}
        placeholders = ",".join("?" * len(doc_ids))
        rows = self._conn.execute(
            f"SELECT doc_id, text FROM docs WHERE doc_id IN ({placeholders})",
            [str(d) for d in doc_ids],
        ).fetchall()
        return {doc_id: text for doc_id, text in rows}

    def close(self) -> None:
        self._conn.close()


_store_cache: dict[str, DocStore] = {}


def get_docstore(dataset: str) -> DocStore:
    if dataset not in _store_cache:
        _store_cache[dataset] = DocStore(dataset)
    return _store_cache[dataset]
