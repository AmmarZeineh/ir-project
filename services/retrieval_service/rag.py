import re
import httpx
from services.retrieval_service.embedder import retrieve_embedding
from services.retrieval_service.retriever import retrieve_bm25
from shared.docstore import get_docstore


def rag_answer(
    query: str,
    query_tokens: list[str],
    dataset: str = "quora",
    top_k: int = 5,
) -> dict:
    bm25_res  = retrieve_bm25(query_tokens, top_k=top_k, dataset=dataset)
    embed_res = retrieve_embedding(query, top_k=top_k, dataset=dataset)

    seen, candidates = set(), []
    for r in bm25_res + embed_res:
        if r["doc_id"] not in seen:
            seen.add(r["doc_id"])
            candidates.append(r)
    candidates = candidates[:top_k]

    docstore = get_docstore(dataset)
    docs_map = docstore.get_many([r["doc_id"] for r in candidates])
    context_parts = []
    for i, r in enumerate(candidates, 1):
        text = docs_map.get(r["doc_id"], "")[:400]
        context_parts.append(f"[Doc {i}] {text}")
    context = "\n\n".join(context_parts)

    prompt = f"""You are an expert information retrieval assistant.
Answer the question based ONLY on the provided documents.
Be concise (2-4 sentences). If the answer is not in the documents, say so.

Question: {query}

Documents:
{context}

Answer:"""

    try:
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:1b", "prompt": prompt, "stream": False},
            timeout=60
        )
        answer = response.json()["response"].strip()
    except Exception:
        query_words = set(query.lower().split())
        best_sentence, best_score = "", 0
        for part in context_parts:
            for sent in re.split(r'[.!?]', part):
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                score = len(query_words & set(sent.lower().split()))
                if score > best_score:
                    best_score = score
                    best_sentence = sent
        answer = best_sentence or "Answer not found in the retrieved documents."

    return {
        "query":   query,
        "answer":  answer,
        "sources": candidates,
        "dataset": dataset,
    }