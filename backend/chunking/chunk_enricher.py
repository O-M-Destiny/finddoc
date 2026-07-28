from chunking import patch_security

import os
import json
import pickle
import time
import random
import sys
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "partitioned_dataset")
CHUNKED_ELEMENTS = os.path.join(CACHE_DIR, "Nvidia-Report25_chunks.pkl")
JSON_DB = os.path.join(BASE_DIR, "json_database", "enriched_summaries.json")

class ChunkEnricher:
    def __init__(self, chunks: list):
        self.chunks = chunks
        self.primary_model_exhausted = False

    def separate_content(self, chunk) -> dict:
        content_data = {
            'text': chunk.text,
            'image': [],
            'tables': [],
            'types': ['text']
        }

        if chunk.category == 'Table':
            content_data['types'] = ['table']
            content_data['tables'].append({
                'text_as_html': getattr(chunk.metadata, 'text_as_html', chunk.text),
                'pdfplumber_extracted': getattr(chunk.metadata, 'pdfplumber_extracted', None)
            })
            return content_data

        if chunk.category == 'Image':
            content_data['types'] = ['image']
            if hasattr(chunk.metadata, 'image_base64'):
                content_data['image'].append(chunk.metadata.image_base64)
            return content_data

        if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements') and chunk.metadata.orig_elements is not None:
            for element in chunk.metadata.orig_elements:
                element_type = type(element).__name__

                if element_type == 'Table':
                    content_data['types'] = ['table']
                    content_data['tables'].append({
                        'text_as_html': getattr(element.metadata, 'text_as_html', element.text),
                        'pdfplumber_extracted': getattr(element.metadata, 'pdfplumber_extracted', None)
                    })

                elif element_type == 'Image':
                    if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                        content_data['types'] = ['image']
                        content_data['image'].append(element.metadata.image_base64)

        return content_data

    def chunk_ai_summary(self, text: str, image: list, tables: list) -> str:

        def load_prompt(filename: str) -> str:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base_dir, "prompts", f"{filename}.txt")
            with open(path, 'r') as f:
                return f.read()

        def build_prompt(text: str, tables: list, images: list) -> str:
            if tables:
                tables_section = "TABLES:\n"
                for i, table in enumerate(tables):
                    tables_section += f"Table {i+1}:\n"
                    tables_section += f"Raw text: {table.get('text_as_html', '')}\n"
                    tables_section += f"pdfplumber: {table.get('pdfplumber_extracted', '')}\n\n"
            else:
                tables_section = ""

            table_instruction: str = load_prompt('table_prompts') if tables else ""
            image_instruction: str = load_prompt('image_prompts') if images else ""

            return load_prompt('base_prompts').format(
                text=text,
                tables_section=tables_section,
                table_instruction=table_instruction,
                image_instruction=image_instruction
            )

        def call_llm(model: str, prompt: str) -> str:
            llm = ChatGroq(model=model, temperature=0)
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = llm.invoke([HumanMessage(content=[{"type": "text", "text": prompt}])])
                    return response.content.strip()
                except Exception as e:
                    error_msg = str(e).lower()
                    if 'rate limit' in error_msg or '429' in error_msg or 'connection error' in error_msg:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limit hit ({model}). Waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}...")
                        time.sleep(wait_time)
                    else:
                        raise e

            raise RuntimeError(f"Hard rate limit reached for model: {model}")

        prompt = build_prompt(text, tables, image)

        # Try Primary Model ONLY if it hasn't hit a hard limit yet in this session
        if not self.primary_model_exhausted:
            try:
                print("Calling primary model...")
                return call_llm("openai/gpt-oss-120b", prompt)
            except RuntimeError:
                print("\n[!] Primary model daily limit reached! Switching to fallback model permanently for this session...\n")
                self.primary_model_exhausted = True  # Remember this for subsequent iterations

        # Try Fallback Model
        print("Calling fallback model...")
        return call_llm("meta-llama/llama-4-scout-17b-16e-instruct", prompt)

    def enrich(self) -> list:
        enriched = []

        if os.path.exists(JSON_DB):
            with open(JSON_DB, 'r') as f:
                enriched = json.load(f)

            if len(enriched) == len(self.chunks):
                print(f"Already complete. Loaded {len(enriched)} enriched chunks")
                return enriched

            print(f"Resuming from chunk {len(enriched) + 1}/{len(self.chunks)}...")
        else:
            print(f"Starting fresh. Processing {len(self.chunks)} chunks...")
            os.makedirs(os.path.dirname(JSON_DB), exist_ok=True)

        total = len(self.chunks)
        start_from = len(enriched)

        for i, chunk in enumerate(self.chunks[start_from:], start=start_from):
            content = self.separate_content(chunk)

            try:
                content['summary'] = self.chunk_ai_summary(
                    text=content['text'],
                    image=content['image'],
                    tables=content['tables']
                )
            except Exception as e:
                print(f"\n[!] Hard Rate Limit Triggered: {e}")
                print("Both models have exhausted their quotas for today.")
                print(f"Progress cleanly saved up to chunk {i}. Exiting gracefully...")
                print("Run this script tomorrow to pick up exactly where you left off!\n")
                sys.exit(0)

            content_to_save = {
                'text': content['text'],
                'tables': content['tables'],
                'types': content['types'],
                'summary': content['summary'],
                'has_image': len(content['image']) > 0
            }
            enriched.append(content_to_save)

            with open(JSON_DB, 'w') as f:
                json.dump(enriched, f, indent=2)

        return enriched





if __name__ == "__main__":
    with open(CHUNKED_ELEMENTS, 'rb') as f:
        chunks = pickle.load(f)

    enricher = ChunkEnricher(chunks=chunks)
    enriched_chunks = enricher.enrich()


