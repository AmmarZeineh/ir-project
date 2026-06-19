import os
import sys
from typing import Optional
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
sys.path.insert(0, os.path.abspath("."))
from shared.docstore import get_docstore
from shared.schemas import SearchRequest, SearchResponse, SearchResult

app = FastAPI(title="IR System — API Gateway", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="services/api_gateway/templates")

SERVICES = {
    "preprocessing":    "http://localhost:8001",
    "retrieval":        "http://localhost:8003",
    "query_refinement": "http://localhost:8005",
}

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    async with httpx.AsyncClient(timeout=30) as client:
        if req.use_refinement:
            ref_res = await client.post(
                f"{SERVICES['query_refinement']}/refine",
                json={"query": req.query, "use_synonyms": True},
            )
            refinement = ref_res.json()
            refined = refinement["expanded"]
        else:
            refined = req.query
            refinement = {
                "original": req.query,
                "corrected": req.query,
                "expanded": req.query,
                "suggestions": [],
                "changed": False,
            }

        pre_res = await client.post(
            f"{SERVICES['preprocessing']}/preprocess",
            json={"text": refined},
        )
        tokens = pre_res.json()["final"]

        ret_res = await client.post(
            f"{SERVICES['retrieval']}/retrieve",
            json={
                "query": refined,
                "tokens": tokens,
                "model": req.model,
                "top_k": req.top_k,
                "k1": req.k1,
                "b": req.b,
                "fusion_method": "rrf",
                "fusion_weights": req.fusion_weights,
                "dataset": req.dataset,
            },
        )
        payload = ret_res.json()
        if "results" not in payload:
            raise HTTPException(status_code=400, detail=payload)
        raw = payload["results"]

    docstore = get_docstore(req.dataset)
    docs_map = docstore.get_many([r["doc_id"] for r in raw])
    results = [
        SearchResult(
            doc_id=r["doc_id"],
            score=r["score"],
            text=docs_map.get(r["doc_id"], ""),
            stage=r.get("stage"),
        )
        for r in raw
    ]
    return SearchResponse(
        query_original=req.query,
        query_refined=refined,
        model=req.model,
        results=results,
        refinement_info=refinement,
    )

class RAGRequest(BaseModel):
    query: str
    dataset: str = "quora"
    top_k: int = 5


@app.post("/rag")
async def rag_search(req: RAGRequest):
    """Proxy endpoint for RAG search.

    This endpoint delegates preprocessing and retrieval to the respective
    services via HTTP requests instead of importing functions directly.
    It first calls the preprocessing service to obtain the tokenized query,
    then calls the retrieval service's `/rag` endpoint to get the final
    answer and sources.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        pre_res = await client.post(
            f"{SERVICES['preprocessing']}/preprocess",
            json={"text": req.query},
        )
        pre_res.raise_for_status()
        tokens = pre_res.json()["final"]
        rag_res = await client.post(
            f"{SERVICES['retrieval']}/rag",
            json={
                "query": req.query,
                "tokens": tokens,
                "dataset": req.dataset,
                "top_k": req.top_k,
            },
        )
        rag_res.raise_for_status()
        return rag_res.json()

@app.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=5) as client:
        statuses = {}
        for name, url in SERVICES.items():
            try:
                r = await client.get(f"{url}/health")
                statuses[name] = r.json()
            except httpx.HTTPError:
                statuses[name] = "unreachable"
    return {"gateway": "ok", "services": statuses}