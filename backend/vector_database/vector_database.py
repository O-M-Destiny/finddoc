import os
import json

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, "json_database", "final_processed_chunks_FIXED.json")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")


class VectorStore:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)

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
        """Builds a fresh FAISS index from the fixed JSON chunks and saves it to disk."""
        documents = self._json_to_documents()
        print(f"Loaded {len(documents)} documents from {JSON_PATH}")

        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model,
        )
        vector_store.save_local(FAISS_INDEX_DIR)
        print(f"FAISS index built and saved to {FAISS_INDEX_DIR}")
        return vector_store

    def load_existing_db(self):
        """Checks if a FAISS index already exists first. Only embeds if it doesn't."""
        if not os.path.exists(FAISS_INDEX_DIR):
            print("No existing FAISS index found — building a new one.")
            return self.build_and_store_db()

        print(f"Existing FAISS index found at {FAISS_INDEX_DIR} — loading it (no re-embedding).")
        vector_store = FAISS.load_local(
            folder_path=FAISS_INDEX_DIR,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True,
        )
        return vector_store


if __name__ == "__main__":
    vstore_manager = VectorStore()
    db = vstore_manager.load_existing_db()