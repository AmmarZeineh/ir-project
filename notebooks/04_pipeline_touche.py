import sys, os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from services.preprocessing_service.preprocessor import preprocess
from services.indexing_service.indexer import (
    build_inverted_index, build_doc_lengths,
    compute_idf, save_index
)
from services.retrieval_service.embedder import build_doc_embeddings

import services.indexing_service.indexer as idx_module
import services.retrieval_service.embedder as emb_module

RAW  = Path("data/raw/touche")
PROC = Path("data/processed/touche")
IDX  = Path("data/indexes/touche")
PROC.mkdir(parents=True, exist_ok=True)
IDX.mkdir(parents=True, exist_ok=True)

print("⚙️  Step 1: Preprocessing docs...")
docs_df = pd.read_csv(RAW / "docs_relevant.csv")
tqdm.pandas(desc="Processing docs")
docs_df["processed"] = docs_df["text"].progress_apply(
    lambda t: preprocess(str(t))["final_str"]
)
docs_df[["doc_id","processed"]].to_csv(PROC / "docs_processed.csv", index=False)
print(f"   ✅ {len(docs_df)} docs processed")

print("⚙️  Processing queries...")
qdf = pd.read_csv(RAW / "queries.csv")
qdf["processed"] = qdf["text"].apply(lambda t: preprocess(str(t))["final_str"])
qdf.to_csv(PROC / "queries_processed.csv", index=False)
print(f"   ✅ {len(qdf)} queries processed")

print("\n🔨 Step 2: Building indexes...")
idx_module.INDEX_DIR = IDX
index    = build_inverted_index(docs_df)
doc_lens = build_doc_lengths(docs_df)
idf      = compute_idf(index, len(docs_df))
save_index(index,    "inverted_index")
save_index(doc_lens, "doc_lengths")
save_index(idf,      "idf")
print(f"   ✅ Vocab: {len(index):,} | Docs: {len(doc_lens):,}")

print("\n🧠 Step 3: Building BERT embeddings...")
emb_module.INDEX_DIR = IDX
build_doc_embeddings(docs_df, batch_size=512)

print("\n✅ Pipeline complete!")
for f in sorted(IDX.iterdir()):
    print(f"   {f.name:<30} {f.stat().st_size/1024/1024:.1f} MB")