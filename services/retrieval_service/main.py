from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from .retriever import retrieve_tfidf, retrieve_bm25
from .embedder  import retrieve_embedding
from .hybrid    import retrieve_hybrid_serial, retrieve_hybrid_parallel
from .rag      import rag_answer

app = FastAPI(title="Retrieval Service", version="1.0")

class RetrievalRequest(BaseModel):
    query: str
    tokens: list[str]
    model: str = "bm25"
    top_k: int = 10
    k1: float = 1.5
    b: float = 0.75
    fusion_method: str = "rrf"
    fusion_weights: Optional[list[float]] = None
    dataset: str = "quora"

@app.post("/retrieve")
def retrieve(body: RetrievalRequest):
    ds = body.dataset
    if body.model == "tfidf":
        results = retrieve_tfidf(body.tokens, body.top_k, dataset=ds)
    elif body.model == "bm25":
        results = retrieve_bm25(body.tokens, body.top_k, body.k1, body.b, dataset=ds)
    elif body.model == "embedding":
        results = retrieve_embedding(body.query, body.top_k, dataset=ds)
    elif body.model.startswith("embedding_"):
        results = retrieve_embedding(body.query, body.top_k, dataset=ds)
    elif body.model == "hybrid_serial":
        results = retrieve_hybrid_serial(
            body.query,
            body.tokens,
            body.top_k,
            k1=body.k1,
            b=body.b,
            dataset=ds,
        )
    elif body.model.startswith("hybrid_parallel"):
        method = "rrf" if "rrf" in body.model else "linear"
        results = retrieve_hybrid_parallel(
            body.query,
            body.tokens,
            body.top_k,
            fusion_method=method,
            weights=body.fusion_weights,
            k1=body.k1,
            b=body.b,
            dataset=ds,
        )
    elif body.model == "rag":
        ans = rag_answer(body.query, body.tokens, dataset=ds, top_k=body.top_k)
        results = ans.get("sources", [])
    else:
        return {"error": f"Unknown model: {body.model}"}
    return {"model": body.model, "results": results}


class RagRequest(BaseModel):
    query: str
    tokens: list[str]
    dataset: str = "quora"
    top_k: int = 5


@app.post("/rag")
def rag_search(body: RagRequest):
    """Endpoint for RAG answers and retrieval.

    This endpoint takes a raw query and its preprocessed tokens and returns
    the generated answer along with the supporting documents. It relies on
    the local ``rag_answer`` function and does not call external services.
    """
    ans = rag_answer(body.query, body.tokens, dataset=body.dataset, top_k=body.top_k)
    return ans

@app.get("/health")
def health():
    return {"status": "ok", "service": "retrieval", "port": 8003}