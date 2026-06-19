from fastapi import FastAPI
from pydantic import BaseModel

from .evaluator import load_qrels, precision_at_k, recall, average_precision, ndcg

app = FastAPI(title="Ranking & Evaluation Service", version="1.0")


def _qrels_path(dataset: str) -> str:
    return f"data/raw/{dataset}/qrels.csv"


class EvalRequest(BaseModel):
    query_id: str
    results: list[dict]
    dataset: str = "quora"

_qrels_cache = {}


def _get_qrels(dataset: str) -> dict:
    if dataset not in _qrels_cache:
        _qrels_cache[dataset] = load_qrels(_qrels_path(dataset))
    return _qrels_cache[dataset]


@app.post("/evaluate")
def evaluate(body: EvalRequest):
    qrels = _get_qrels(body.dataset)
    return {
        "query_id": body.query_id,
        "dataset": body.dataset,
        "P@10":     round(precision_at_k(body.results, qrels, body.query_id), 4),
        "Recall":   round(recall(body.results, qrels, body.query_id), 4),
        "AP":       round(average_precision(body.results, qrels, body.query_id), 4),
        "nDCG@10":  round(ndcg(body.results, qrels, body.query_id), 4),
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "ranking", "port": 8004}
