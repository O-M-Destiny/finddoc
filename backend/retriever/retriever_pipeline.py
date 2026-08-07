import os
import json
import re
from dotenv import load_dotenv

from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.documents import Document

# from sentence_transformers import CrossEncoder

from .fusion.rrf import reciprocal_rank_fusion
from ..vector_database.vector_database import VectorStore
from ..pydantic_schema.structured_model_output import generate_queries

load_dotenv()

# os.environ['HF_HUB_OFFLINE'] = "1"

# CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "json_database", "final_processed_chunks_FIXED.json")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "json_database", "final_processed_chunks_FIXED.json")



class RAGRetriever:
    def __init__(self, llm_model: str , model_name: str = "BAAI/bge-large-en-v1.5", chunks_path: str = CHUNKS_PATH):
        self.vstore_manager = VectorStore(model_name=model_name)
        self.db: QdrantVectorStore = self.vstore_manager.load_existing_db()
        self.llm = ChatGroq(model=llm_model)
        # self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        self.bm25_retriever = None
        self._init_bm25(chunks_path)
    
    def _init_bm25(self, chunks_path: str) -> None:
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            docs = [
                Document(
                    page_content=chunk["page_content"],metadata=chunk.get("metadata", {})
                )
                for chunk in chunks.values()
            ]
            if not docs:
                return
            self.bm25_retriever = BM25Retriever.from_documents(docs)

        except Exception as e:
            print(f"Error initializing BM25 retriever: {e}")
            self.bm25_retriever = None
    
    
    

    def _format_documents(self, docs: list[Document]) -> list[dict]:
        """This Method is for printing to the developer i.e ME! lol"""
        formatted_results = []

        for rank, doc in enumerate(docs, start=1):
            types = doc.metadata.get("types_found", ["text"])

            formatted_results.append({
                "rank": rank,
                "content": doc.page_content,
                "ai_summary": doc.metadata.get(
                    "ai_summary",
                    "No summary available."
                ),
                "types_found": ", ".join(types) if isinstance(types, list) else str(types),
                "has_image": doc.metadata.get("has_image", False)
            })

        return formatted_results


    def _normalize_query(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+([?.!,:;])", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def generate_query_variations(self, question: str, n: int = 3) -> list[str]:
        queries = generate_queries(question=question, llm=self.llm, n=n)
        queries.append(question)

        seen = set()
        unique_queries = []

        for q in queries:
            normalized = self._normalize_query(q)
            if normalized not in seen:
                seen.add(normalized)
                unique_queries.append(q)

        return unique_queries
    
    def _dense_document_search(self, queries: list[str], k: int) -> list[list[Document]]:
        result_lists = []
        for q in queries:
            try:
                results = self.db.similarity_search(q, k=k)
            except Exception as e:
                print(f'Error searching for query {q} : {e}')
                continue
            result_lists.append(results)
        return result_lists

    
    def _bm25_documents_search(self, queries: list[str], k: int) -> list[list[Document]]:
        if self.bm25_retriever is None:
            return []

        self.bm25_retriever.k = k
        result_lists = []
        for q in queries:
            try:
                results = self.bm25_retriever.invoke(q)
            except Exception as e:
                print(f"Error in BM25 search for query '{q}': {e}")
                continue
            result_lists.append(results)
        return result_lists

    
    def hybrid_retriever_with_queries(self, queries: list[str], k: int = 4 ) -> list[Document]:
        """ k = 4 is just for testing, ideally i would increase it to 8 or more. I choosed 4 because i am using a free
            llm and the context window will be limited"""

        bm25_result_lists = self._bm25_documents_search(queries=queries, k=k)
        dense_result_lists = self._dense_document_search(queries=queries, k=k)

        # Stage 1: fuse each retriever's own multi-query results into one honest ranking
        bm25_fused = reciprocal_rank_fusion(bm25_result_lists)
        dense_fused = reciprocal_rank_fusion(dense_result_lists)

        # Stage 2: fuse the two retrievers together, same as before
        fused = reciprocal_rank_fusion([bm25_fused, dense_fused])
        return fused

    def hybrid_retriever(self, question: str, n: int = 3, k: int = 4) -> list[Document]:
        queries = self.generate_query_variations(question=question, n=n)
        print(queries)
        return self.hybrid_retriever_with_queries(queries=queries, k=k)

    
    
    # def _rerank_documents(self, queries: list[str], documents: list[Document], top_n: int = 8) -> list[Document]:
    #     """Ignore this method for now. The re-ranker is not yet giving the result i want. I will 
    #     skip it for now"""

    #     if not documents:
    #         return []

    #     ranked_lists = []
    #     for q in queries:
    #         pairs = [(q, doc.page_content) for doc in documents]
    #         scores = self.reranker.predict(pairs)

    #         scored_docs = list(zip(documents, scores))
    #         scored_docs.sort(key=lambda x: x[1], reverse=True)

    #         ranked_lists.append([doc for doc, score in scored_docs])

    #     fused = reciprocal_rank_fusion(ranked_lists)
    #     return fused[:top_n]
        

if __name__ == "__main__":
    retriever = RAGRetriever(llm_model="openai/gpt-oss-120b")

    queries = retriever.generate_query_variations("Who is the founder of Nvidia?")
    print("Queries:", queries)

    docs = retriever.hybrid_retriever_with_queries(queries=queries, k=3)
    print(f"Retrieved {len(docs)} documents")
    