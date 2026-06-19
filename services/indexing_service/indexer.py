import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

INDEX_DIR = Path("data/indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_index_dir(index_dir: Path | str | None = None) -> Path:
    path = Path(index_dir) if index_dir is not None else INDEX_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_inverted_index(docs_df: pd.DataFrame) -> dict:
    """
    بيبني Inverted Index من الـ docs المعالجة.
    الشكل: { term: { doc_id: term_frequency } }
    """
    index = defaultdict(dict)

    for _, row in docs_df.iterrows():
        doc_id = str(row["doc_id"])
        tokens = str(row["processed"]).split()
        for term, freq in Counter(tokens).items():
            index[term][doc_id] = freq

    return dict(index)


def build_doc_lengths(docs_df: pd.DataFrame) -> dict:
    """طول كل doc (عدد الكلمات بعد المعالجة) — مطلوب لـ BM25"""
    return {
        str(row["doc_id"]): len(str(row["processed"]).split())
        for _, row in docs_df.iterrows()
    }


def compute_idf(index: dict, num_docs: int) -> dict:
    """IDF لكل term"""
    return {
        term: math.log((num_docs - len(postings) + 0.5) / (len(postings) + 0.5) + 1)
        for term, postings in index.items()
    }


def save_index(index: dict, name: str, index_dir: Path | str | None = None) -> None:
    path = _resolve_index_dir(index_dir) / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(index, f)
    size_kb = path.stat().st_size / 1024
    print(f"   💾 Saved {name}.pkl  ({size_kb:.1f} KB)")


def load_index(name: str, index_dir: Path | str | None = None) -> Any:
    path = _resolve_index_dir(index_dir) / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
