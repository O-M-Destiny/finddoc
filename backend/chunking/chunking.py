import os
import pickle
import json

from unstructured.chunking.title import chunk_by_title

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "partitioned_dataset")
CLEANED_CACHE = os.path.join(CACHE_DIR, "Nvidia-Report25_cleaned.pkl")

with open(CLEANED_CACHE, 'rb') as f:
    cleaned_elements = pickle.load(f)


class Chunk:
    def __init__(self, elements: list):
        self.elements = elements

    def create_chunk_by_title(self, elements: list = None) -> list:
        target_elements = elements if elements is not None else self.elements

        chunks = chunk_by_title(
            elements=target_elements,
            max_characters=2600,
            new_after_n_chars=2200,
            overlap=100,
            combine_text_under_n_chars=200
        )

        return chunks
    
    def finalized_chunks(self) -> list:
        final_chunks: list = []
        text_buffer: list = []
        
        for el in self.elements:
            if el.category in ['Image', 'Table']:
                
                if text_buffer:
                    chunked_text = self.create_chunk_by_title(elements=text_buffer)
                    final_chunks.extend(chunked_text)
                    text_buffer = [] 
                
                final_chunks.append(el)
            else:
                text_buffer.append(el)
        
        if text_buffer:
            chunked_text = self.create_chunk_by_title(elements=text_buffer)
            final_chunks.extend(chunked_text)
            
        print(f"Created {len(final_chunks)} Total Processed Pipeline Elements")
        return final_chunks
    
    def save_chunks(self, chunks: list, filename: str = "Nvidia-Report25_chunks.pkl"):
        output_path = os.path.join(CACHE_DIR, filename)
        
        if os.path.exists(output_path):
            print(f"Loading chunks from cache: {output_path}")
            with open(output_path, 'rb') as f:
                return pickle.load(f)
        
        with open(output_path, 'wb') as f:
            pickle.dump(chunks, f)
        print(f"Saved {len(chunks)} chunks ")
        return chunks
    


chunker = Chunk(elements=cleaned_elements)
chunks = chunker.finalized_chunks()
chunks = chunker.save_chunks(chunks)
