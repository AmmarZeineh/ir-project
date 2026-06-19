import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from shared.data_paths import index_dataset_dir

"""Embedding utilities for the retrieval service.

This module provides a single SentenceTransformer model and a set of
precomputed document embeddings for that model.  It exposes helper
functions for computing and loading the embeddings and for retrieving
documents using cosine similarity.  Unlike previous iterations of this
project, this version supports **only one** embedding model.  The
ability to select between multiple models or fuse their outputs has
been removed to simplify the system and meet the base project
requirements.
"""

MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_DIR = Path("data/indexes")

_model: SentenceTransformer | None = None
_embed_cache: dict[str, tuple[np.ndarray, list[str]]] = {}


def _index_dir_for_dataset(dataset: str) -> Path:
    """Return the index directory for a given dataset."""
    return index_dataset_dir(dataset)


def get_model() -> SentenceTransformer:
    """Return a cached SentenceTransformer model.

    The model is loaded the first time this function is called and
    reused on subsequent calls.  Only a single model is supported.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_embeddings(dataset: str = "quora") -> tuple[np.ndarray, list[str]]:
    """Return document embeddings and document IDs for a dataset.

    Embeddings are cached on first access.  They are loaded from
    ``doc_embeddings.npy`` and ``doc_ids_embed.pkl`` within the
    dataset's index directory.
    """
    if dataset not in _embed_cache:
        idx_dir = _index_dir_for_dataset(dataset)
        embeddings = np.load(idx_dir / "doc_embeddings.npy")
        with open(idx_dir / "doc_ids_embed.pkl", "rb") as f:
            doc_ids = pickle.load(f)
        _embed_cache[dataset] = (embeddings, doc_ids)
    return _embed_cache[dataset]


def build_doc_embeddings(docs_df, batch_size: int = 256, index_dir: str | Path | None = None) -> tuple[np.ndarray, list[str]]:
    """Compute and persist document embeddings using the single model.

    Parameters
    ----------
    docs_df : pandas.DataFrame
        DataFrame containing ``processed`` and ``doc_id`` columns.
    batch_size : int, optional
        Batch size for encoding. Defaults to 256.
    index_dir : str or Path, optional
        Directory in which to write the embeddings.  Defaults to
        ``INDEX_DIR``.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        The computed embeddings and corresponding document IDs.
    """
    model  = get_model()
    texts  = docs_df["processed"].astype(str).tolist()
    doc_ids = docs_df["doc_id"].astype(str).tolist()
    idx_dir = Path(index_dir) if index_dir is not None else INDEX_DIR
    idx_dir.mkdir(parents=True, exist_ok=True)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(idx_dir / "doc_embeddings.npy", embeddings)
    with open(idx_dir / "doc_ids_embed.pkl", "wb") as f:
        pickle.dump(doc_ids, f)
    print(f"   ✅ Saved embeddings shape: {embeddings.shape}")
    return embeddings, doc_ids


def retrieve_embedding(query: str, top_k: int = 10, dataset: str = "quora") -> list[dict]:
    """Retrieve documents using the single embedding model.

    Parameters
    ----------
    query : str
        Raw query string.
    top_k : int, optional
        Number of results to return. Defaults to 10.
    dataset : str, optional
        Dataset name. Defaults to ``"quora"``.

    Returns
    -------
    list[dict]
        A list of dictionaries with ``doc_id`` and ``score``.
    """
    model = get_model()
    embeddings, doc_ids = _get_embeddings(dataset)
    q_vec  = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ q_vec
    top_i  = np.argsort(scores)[::-1][:top_k]
    return [
        {"doc_id": doc_ids[i], "score": round(float(scores[i]), 4)}
        for i in top_i
    ]


