import os
import json
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, "json_database", "final_processed_chunks_FIXED.json")

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
COLLECTION_NAME = "finddoc_chunks"


class VectorStore:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.embedding_model = HuggingFaceEndpointEmbeddings(
            model=self.model_name,
            huggingfacehub_api_token=HF_TOKEN,
        )
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    def _json_to_documents(self, path: str = JSON_PATH) -> list[Document]:
        """Reads the fixed chunk JSON and turns it into LangChain Documents, in order."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        keys_in_order = sorted(data.keys(), key=lambda k: int(k))

        documents = []
        for key in keys_in_order:
            chunk = data[key]
            documents.append(
                Document(
                    page_content=chunk["page_content"],
                    metadata=chunk.get("metadata", {}),
                )
            )
        return documents

    def build_and_store_db(self):
        """Builds fresh vectors from the fixed JSON chunks and uploads them to Qdrant."""
        documents = self._json_to_documents()
        print(f"Loaded {len(documents)} documents from {JSON_PATH}")

        vector_store = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
            timeout=60,
        )
        print(f"Uploaded {len(documents)} chunks to Qdrant collection '{COLLECTION_NAME}'")
        return vector_store

    def load_existing_db(self):
        """Checks if the Qdrant collection already exists. Only embeds+uploads if it doesn't."""
        collections = [c.name for c in self.client.get_collections().collections]

        if COLLECTION_NAME not in collections:
            print(f"No existing Qdrant collection '{COLLECTION_NAME}' found — building a new one.")
            return self.build_and_store_db()

        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embedding_model,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
        )
        return vector_store


if __name__ == "__main__":
    vstore_manager = VectorStore()
    db = vstore_manager.load_existing_db()