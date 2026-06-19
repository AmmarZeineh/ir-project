import numpy as np
from services.retrieval_service.retriever import retrieve_bm25, retrieve_tfidf
from services.retrieval_service.embedder  import retrieve_embedding


def retrieve_hybrid_serial(
    query: str,
    query_tokens: list[str],
    top_k: int = 10,
    first_stage_k: int = 100,
    k1: float = 1.5,
    b: float = 0.75,
    dataset: str = "quora",
) -> list[dict]:
    """Retrieve documents using BM25 followed by embedding re-ranking.

    This function implements a two-stage retrieval where BM25 is used to
    select a set of candidate documents (``first_stage_k``) and then an
    embedding-based similarity is used to re-rank those candidates.  It
    uses the single embedding model defined in ``embedder.py``.  The
    ability to choose a different embedding model has been removed.
    """
    bm25_results   = retrieve_bm25(query_tokens, top_k=first_stage_k, k1=k1, b=b, dataset=dataset)
    candidate_ids  = [r["doc_id"] for r in bm25_results]
    if not candidate_ids:
        return []

    from services.retrieval_service.embedder import get_model, _get_embeddings
    model = get_model()
    doc_embeddings, doc_ids = _get_embeddings(dataset)

    id_to_idx = {did: i for i, did in enumerate(doc_ids)}
    filtered_ids = [cid for cid in candidate_ids if cid in id_to_idx]
    cand_idx  = [id_to_idx[cid] for cid in filtered_ids]
    if not cand_idx:
        return bm25_results[:top_k]

    cand_emb = doc_embeddings[cand_idx]
    q_vec    = model.encode([query], normalize_embeddings=True)[0]
    scores   = cand_emb @ q_vec

    ranked = np.argsort(scores)[::-1][:top_k]
    return [
        {"doc_id": filtered_ids[i], "score": round(float(scores[i]), 4), "stage": "serial"}
        for i in ranked
    ]


def _rrf(results_list, k=60):
    rrf_scores = {}
    for results in results_list:
        for rank, item in enumerate(results, start=1):
            doc_id = item["doc_id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"doc_id": d, "score": round(s, 6), "stage": "parallel_rrf"} for d, s in ranked]


def _linear(results_list, weights=None):
    if weights is None:
        weights = [1.0] * len(results_list)
    combined = {}
    for w, results in zip(weights, results_list):
        if not results:
            continue
        max_s = max(r["score"] for r in results)
        min_s = min(r["score"] for r in results)
        denom = max_s - min_s if max_s != min_s else 1
        for item in results:
            doc_id = item["doc_id"]
            norm   = (item["score"] - min_s) / denom
            combined[doc_id] = combined.get(doc_id, 0) + w * norm
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [{"doc_id": d, "score": round(s, 6), "stage": "parallel_linear"} for d, s in ranked]


def retrieve_hybrid_parallel(
    query: str,
    query_tokens: list[str],
    top_k: int = 10,
    fusion_method: str = "rrf",
    weights=None,
    k1: float = 1.5,
    b: float = 0.75,
    dataset: str = "quora",
) -> list[dict]:
    """Retrieve documents using BM25, TF-IDF and embeddings in parallel.

    This function computes BM25, TF-IDF and embedding rankings in parallel
    and then fuses them using either reciprocal rank fusion (RRF) or a
    linear combination.  Support for multiple embedding models has
    been removed; it always uses the single embedding model defined in
    ``embedder.py``.
    """
    n = top_k * 3
    results_list = [
        retrieve_bm25(query_tokens, top_k=n, k1=k1, b=b, dataset=dataset),
        retrieve_tfidf(query_tokens, top_k=n, dataset=dataset),
        retrieve_embedding(query, top_k=n, dataset=dataset),
    ]
    fused = _rrf(results_list) if fusion_method == "rrf" else _linear(results_list, weights)
    return fused[:top_k]
