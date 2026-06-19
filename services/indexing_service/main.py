from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

from .indexer import build_inverted_index, build_doc_lengths, compute_idf, save_index
from shared.data_paths import index_dataset_dir, processed_dataset_dir

app = FastAPI(title="Indexing Service", version="1.0")


class BuildRequest(BaseModel):
    dataset: str = "quora"


@app.post("/build")
def build(body: BuildRequest | None = None):
    dataset = body.dataset if body else "quora"
    docs_path = processed_dataset_dir(dataset) / "docs_processed.csv"
    index_dir = index_dataset_dir(dataset)

    docs_df  = pd.read_csv(docs_path)
    index    = build_inverted_index(docs_df)
    doc_lens = build_doc_lengths(docs_df)
    idf      = compute_idf(index, len(docs_df))
    save_index(index,    "inverted_index", index_dir=index_dir)
    save_index(doc_lens, "doc_lengths", index_dir=index_dir)
    save_index(idf,      "idf", index_dir=index_dir)
    return {
        "status": "ok",
        "dataset": dataset,
        "vocab_size": len(index),
        "num_docs": len(docs_df),
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "indexing", "port": 8002}
