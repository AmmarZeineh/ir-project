import sys, os
sys.path.insert(0, os.path.abspath("."))

from shared.data_paths import raw_docs_path
from shared.docstore import build_docstore, docstore_path

DATASETS = ["quora", "touche"]

print("📦 Building SQLite docstore (raw text by doc_id)...\n")

for dataset in DATASETS:
    docs_csv = raw_docs_path(dataset)
    print(f"  ⏳ {dataset}: loading {docs_csv} ...")
    count = build_docstore(dataset, docs_csv)
    db_path = docstore_path(dataset)
    size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"     ✅ {count:,} docs → {db_path} ({size_mb:.1f} MB)\n")

print("✅ Docstore build complete!")
print("   Retrieval/RAG services will now read raw document text from SQLite by doc_id.")
