import math
import pandas as pd
from collections import defaultdict


def load_qrels(path: str = "data/raw/qrels.csv") -> dict:
    """{ query_id: { doc_id: relevance } }"""
    df = pd.read_csv(path)
    qrels = defaultdict(dict)
    for _, row in df.iterrows():
        qrels[str(row["query_id"])][str(row["doc_id"])] = int(row["relevance"])
    return dict(qrels)


def precision_at_k(results: list[dict], qrels: dict, query_id: str, k: int = 10) -> float:
    relevant = qrels.get(query_id, {})
    top_k    = results[:k]
    hits     = sum(1 for r in top_k if relevant.get(r["doc_id"], 0) > 0)
    return hits / k if k > 0 else 0.0


def recall(results: list[dict], qrels: dict, query_id: str) -> float:
    relevant   = {did for did, rel in qrels.get(query_id, {}).items() if rel > 0}
    if not relevant:
        return 0.0
    retrieved  = {r["doc_id"] for r in results}
    return len(relevant & retrieved) / len(relevant)


def average_precision(results: list[dict], qrels: dict, query_id: str) -> float:
    relevant = {did for did, rel in qrels.get(query_id, {}).items() if rel > 0}
    if not relevant:
        return 0.0

    hits, score = 0, 0.0
    for rank, r in enumerate(results, start=1):
        if r["doc_id"] in relevant:
            hits  += 1
            score += hits / rank
    return score / len(relevant)

def ndcg(results: list[dict], qrels: dict, query_id: str, k: int = 10) -> float:
    rel_map = qrels.get(query_id, {})

    def dcg(ranking):
        return sum(
            (2 ** rel_map.get(r["doc_id"], 0) - 1) / math.log2(i + 2)
            for i, r in enumerate(ranking[:k])
        )

    ideal_docs = sorted(rel_map.items(), key=lambda x: x[1], reverse=True)
    ideal      = [{"doc_id": d} for d, _ in ideal_docs[:k]]
    idcg       = dcg(ideal)
    return dcg(results) / idcg if idcg > 0 else 0.0


def evaluate_model(
    retrieve_fn,          
    queries_df: pd.DataFrame,
    qrels: dict,
    model_name: str = "model",
    top_k: int = 10,
) -> pd.DataFrame:
    rows = []
    for _, row in queries_df.iterrows():
        qid     = str(row["query_id"])
        tokens  = str(row["processed"]).split()

        results = retrieve_fn(tokens)

        rows.append({
            "query_id":    qid,
            "query":       row["text"],
            "P@10":        round(precision_at_k(results, qrels, qid, k=10), 4),
            "Recall":      round(recall(results, qrels, qid), 4),
            "AP":          round(average_precision(results, qrels, qid), 4),
            "nDCG@10":     round(ndcg(results, qrels, qid, k=10), 4),
        })

    df = pd.DataFrame(rows)
    df.loc["MEAN"] = df[["P@10", "Recall", "AP", "nDCG@10"]].mean().round(4)
    df.at["MEAN", "query_id"] = "MEAN"
    df.at["MEAN", "query"]    = f"── {model_name} ──"
    return df