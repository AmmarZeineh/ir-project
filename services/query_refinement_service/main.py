from fastapi import FastAPI
from pydantic import BaseModel
from .refiner import refine_query, suggest_from_history

app = FastAPI(title="Query Refinement Service", version="1.0")

class QueryInput(BaseModel):
    query: str
    use_synonyms: bool = True

@app.post("/refine")
def refine(body: QueryInput):
    return refine_query(body.query, body.use_synonyms)

@app.get("/suggest")
def suggest(q: str, top_n: int = 5):
    return {"suggestions": suggest_from_history(q, top_n)}

@app.get("/health")
def health():
    return {"status": "ok", "service": "query_refinement", "port": 8005}