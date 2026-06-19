import sys, os
sys.path.insert(0, os.path.abspath("."))

import ir_datasets
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW = Path("data/raw/quora")
RAW.mkdir(parents=True, exist_ok=True)

dataset = ir_datasets.load("beir/quora/test")

print("💾 Saving queries...")
queries = [{"query_id": q.query_id, "text": q.text} for q in dataset.queries_iter()]
pd.DataFrame(queries).to_csv(RAW / "queries.csv", index=False)
print(f"   ✅ {len(queries)} queries")

print("💾 Saving qrels...")
qrels = [
    {"query_id": q.query_id, "doc_id": q.doc_id, "relevance": q.relevance}
    for q in dataset.qrels_iter()
]
pd.DataFrame(qrels).to_csv(RAW / "qrels.csv", index=False)
print(f"   ✅ {len(qrels)} qrels")

print("💾 Saving ALL docs (522,931)...")
docs = []
for doc in tqdm(dataset.docs_iter(), total=522931, desc="Loading"):
    docs.append({"doc_id": doc.doc_id, "text": doc.text})

pd.DataFrame(docs).to_csv(RAW / "docs_relevant.csv", index=False)
print(f"   ✅ {len(docs)} docs saved")

print("\n📁 data/raw/quora/")
for f in sorted(RAW.iterdir()):
    print(f"   {f.name:<25} {f.stat().st_size/1024:.1f} KB")