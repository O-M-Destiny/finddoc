from langchain_core.documents import Document

def reciprocal_rank_fusion(result_lists: list[list[Document]],k_constant: int = 60,top_k: int | None = None,) -> list[Document]:
    scores: dict[str, float] = {}
    doc_lookup: dict[str, Document] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            key = doc.page_content   # or better: doc.metadata["chunk_id"]

            scores[key] = scores.get(key, 0.0) + 1 / (k_constant + rank)

            if key not in doc_lookup:
                doc_lookup[key] = doc

    ranked = sorted(scores.items(),key=lambda x: x[1],reverse=True,)

    if top_k is not None:
        ranked = ranked[:top_k]

    return [doc_lookup[key] for key, _ in ranked]