import sys, os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from pathlib import Path
import services.indexing_service.indexer   as idx_module
import services.retrieval_service.retriever as ret_module
import services.retrieval_service.embedder  as emb_module

IDX_DIR = Path("data/indexes/touche")
idx_module.INDEX_DIR = IDX_DIR
emb_module.INDEX_DIR = IDX_DIR



from services.retrieval_service.retriever import retrieve_tfidf, retrieve_bm25
from services.retrieval_service.embedder  import retrieve_embedding
from services.retrieval_service.hybrid    import retrieve_hybrid_serial, retrieve_hybrid_parallel
from services.retrieval_service.rag       import rag_answer
from services.ranking_service.evaluator   import load_qrels, evaluate_model

queries_df = pd.read_csv("data/processed/touche/queries_processed.csv")
qrels      = load_qrels("data/raw/touche/qrels.csv")

queries_with_qrels = set(qrels.keys())
queries_df = queries_df[queries_df["query_id"].astype(str).isin(queries_with_qrels)]
print(f"📊 Evaluating on {len(queries_df)} queries from Touche-2020\n")

TOPK = 10

def fn_tfidf(tokens):      return retrieve_tfidf(tokens, top_k=TOPK, dataset="touche")
def fn_bm25(tokens):       return retrieve_bm25(tokens, top_k=TOPK, k1=1.5, b=0.75, dataset="touche")
def fn_embed(tokens):
    return retrieve_embedding(" ".join(tokens), top_k=TOPK, dataset="touche")
def fn_serial(tokens):     return retrieve_hybrid_serial(" ".join(tokens), tokens, top_k=TOPK, dataset="touche")
def fn_parallel(tokens):
    return retrieve_hybrid_parallel(
        " ".join(tokens), tokens, top_k=TOPK, fusion_method="rrf", dataset="touche"
    )

def fn_parallel_linear(tokens):
    return retrieve_hybrid_parallel(
        " ".join(tokens), tokens, top_k=TOPK, fusion_method="linear", dataset="touche"
    )

def fn_rag(tokens):
    q_str = " ".join(tokens)
    ans = rag_answer(q_str, tokens, dataset="touche", top_k=TOPK)
    sources = ans.get("sources", [])
    return [
        {"doc_id": s["doc_id"], "score": s.get("score", 1.0)}
        for s in sources
    ]

def fn_bm25_refined(tokens):
    from services.query_refinement_service.refiner import refine_query
    from services.preprocessing_service.preprocessor import preprocess
    q_str = " ".join(tokens)
    refined = refine_query(q_str, use_synonyms=True)["expanded"]
    new_tokens = preprocess(refined)["final"]
    return retrieve_bm25(new_tokens, top_k=TOPK, k1=1.5, b=0.75, dataset="touche")

models = {
    "TF-IDF":                fn_tfidf,
    "BM25":                  fn_bm25,
    "Embedding":             fn_embed,
    "Hybrid Serial":         fn_serial,
    "Hybrid Parallel (RRF)": fn_parallel,
    "Hybrid Parallel (Linear)": fn_parallel_linear,
    "BM25 (Refined)":        fn_bm25_refined,
    "RAG Top Docs":          fn_rag,
}

summary_rows = []
for name, fn in models.items():
    print(f"  ⏳ {name}...")
    df = evaluate_model(fn, queries_df, qrels, model_name=name, top_k=TOPK)
    mean = df.loc["MEAN"]
    summary_rows.append({
        "Model":   name,
        "MAP":     mean["AP"],
        "P@10":    mean["P@10"],
        "Recall":  mean["Recall"],
        "nDCG@10": mean["nDCG@10"],
    })

summary = pd.DataFrame(summary_rows).sort_values("MAP", ascending=False)
print("\n" + "═"*60)
print("  📊 EVALUATION SUMMARY — Touche-2020")
print("═"*60)
print(summary.to_string(index=False))
print("═"*60)

summary.to_csv("data/processed/touche/evaluation_results.csv", index=False)
print("\n✅ Saved → data/processed/touche/evaluation_results.csv")