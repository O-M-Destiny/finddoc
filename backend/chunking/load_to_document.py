import os
import json
import pickle
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "partitioned_dataset")
CHUNKED_ELEMENTS = os.path.join(CACHE_DIR, "Nvidia-Report25_chunks.pkl")
JSON_DB_DIR = os.path.join(BASE_DIR, "json_database")
JSON_DB = os.path.join(JSON_DB_DIR, "enriched_summaries.json")
EXPORT_JSON = os.path.join(JSON_DB_DIR, "final_processed_chunks.json")


def extract_base64_from_pickle_chunk(chunk) -> list:
    images = []
    
    if getattr(chunk, 'category', None) == 'Image':
        if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'image_base64'):
            images.append(chunk.metadata.image_base64)
        return images

    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements') and chunk.metadata.orig_elements is not None:
        for element in chunk.metadata.orig_elements:
            if type(element).__name__ == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    images.append(element.metadata.image_base64)
                    
    return images


def load_hybrid_data_to_langchain() -> list:
    """Merges local JSON text/summaries with Pickle Base64 strings into LangChain Documents."""
    if not os.path.exists(JSON_DB) or not os.path.exists(CHUNKED_ELEMENTS):
        return []

    with open(JSON_DB, 'r') as f:
        cached_data = json.load(f)

    with open(CHUNKED_ELEMENTS, 'rb') as f:
        pickle_chunks = pickle.load(f)

    langchain_documents = []
    
    for i, item in enumerate(cached_data):
        corresponding_pickle_chunk = pickle_chunks[i]
        
        content_pieces = []
        if item.get('text'):
            content_pieces.append(item['text'])
            
        for table in item.get('tables', []):
            if table.get('text_as_html'):
                content_pieces.append(f"[Source Table HTML]:\n{table['text_as_html']}")
            if table.get('pdfplumber_extracted'):
                content_pieces.append(f"[Source Table Text]:\n{table['pdfplumber_extracted']}")
                
        final_page_content = "\n\n".join(content_pieces)

        base64_images = []
        if item.get('has_image', False):
            base64_images = extract_base64_from_pickle_chunk(corresponding_pickle_chunk)
            final_page_content += f"\n\n[Associated Image Reference: {len(base64_images)} image(s) attached in metadata]"

        doc = Document(
            page_content=final_page_content,
            metadata={
                "ai_summary": item.get('summary', ''),  
                "types_found": item.get('types', ['text']),
                "has_image": item.get('has_image', False),
                "images_base64": base64_images,
                "page_number": item.get('page_number'),
                "filename": item.get('filename'),
            }
        )
        langchain_documents.append(doc)

    return langchain_documents


def export_langchain_docs_to_json(langchain_docs, export_path: str) -> dict:
    """Flattens LangChain Documents into an outer-key indexed JSON dictionary on disk."""
    export_data = {}
    
    for i, doc in enumerate(langchain_docs):
        chunk_key = str(i + 1)
        
        export_data[chunk_key] = {
            "page_content": doc.page_content,
            "metadata": {
                "ai_summary": doc.metadata.get("ai_summary", ""),
                "types_found": doc.metadata.get("types_found", ["text"]),
                "has_image": doc.metadata.get("has_image", False),
                "images_base64": doc.metadata.get("images_base64", []),
                "page_number": doc.metadata.get("page_number"),
                "filename": doc.metadata.get("filename"),
            }
        }
    
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    return export_data


if __name__ == "__main__":
    langchain_documents = load_hybrid_data_to_langchain()

    if langchain_documents:
        export_langchain_docs_to_json(
            langchain_docs=langchain_documents, 
            export_path=EXPORT_JSON
        )