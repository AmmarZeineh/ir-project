import math
from collections import defaultdict
from pathlib import Path

from services.indexing_service.indexer import load_index
from shared.data_paths import index_dataset_dir

_cache = {}


def _index_dir_for_dataset(dataset: str) -> Path:
    return index_dataset_dir(dataset)


def _get_indexes(dataset: str = "quora"):
    if dataset not in _cache:
        idx_dir = _index_dir_for_dataset(dataset)
        index    = load_index("inverted_index", index_dir=idx_dir)
        doc_lens = load_index("doc_lengths", index_dir=idx_dir)
        idf      = load_index("idf", index_dir=idx_dir)

        num_docs = len(doc_lens)
        avg_dl   = sum(doc_lens.values()) / num_docs

        _cache[dataset] = {
            "index": index, "doc_lens": doc_lens,
            "idf": idf, "num_docs": num_docs, "avg_dl": avg_dl
        }
    return _cache[dataset]


def retrieve_tfidf(query_tokens: list[str], top_k: int = 10, dataset: str = "quora") -> list[dict]:
    ctx = _get_indexes(dataset)
    scores = defaultdict(float)

    for term in query_tokens:
        if term not in ctx["index"]:
            continue
        idf_val = ctx["idf"].get(term, 0)
        for doc_id, tf in ctx["index"][term].items():
            tf_weight = (1 + math.log(tf)) if tf > 0 else 0
            scores[doc_id] += tf_weight * idf_val

    for doc_id in scores:
        dl = ctx["doc_lens"].get(doc_id, 1)
        scores[doc_id] /= math.sqrt(dl)

    return _rank(scores, top_k)


def retrieve_bm25(
    query_tokens: list[str], top_k: int = 10,
    k1: float = 1.5, b: float = 0.75, dataset: str = "quora"
) -> list[dict]:
    ctx = _get_indexes(dataset)
    scores = defaultdict(float)

    for term in query_tokens:
        if term not in ctx["index"]:
            continue
        idf_val = ctx["idf"].get(term, 0)
        for doc_id, tf in ctx["index"][term].items():
            dl   = ctx["doc_lens"].get(doc_id, 1)
            norm = 1 - b + b * (dl / ctx["avg_dl"])
            bm25_tf = (tf * (k1 + 1)) / (tf + k1 * norm)
            scores[doc_id] += idf_val * bm25_tf

    return _rank(scores, top_k)


def _rank(scores: dict, top_k: int) -> list[dict]:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"doc_id": doc_id, "score": round(score, 4)} for doc_id, score in ranked[:top_k]]
