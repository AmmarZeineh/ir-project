from pydantic import BaseModel
from typing import Optional

class SearchRequest(BaseModel):
    query: str
    dataset: str = "quora"
    model: str = "bm25"
    top_k: int = 10
    k1: float = 1.5
    b: float = 0.75
    use_refinement: bool = True
    fusion_weights: Optional[list[float]] = None

class SearchResult(BaseModel):
    doc_id: str
    score: float
    text: Optional[str] = None
    stage: Optional[str] = None

class SearchResponse(BaseModel):
    query_original: str
    query_refined: str
    model: str
    results: list[SearchResult]
    refinement_info: Optional[dict] = None