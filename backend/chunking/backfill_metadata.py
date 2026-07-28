import os
import json
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "partitioned_dataset")
CHUNKED_ELEMENTS = os.path.join(CACHE_DIR, "Nvidia-Report25_chunks.pkl")
JSON_DB = os.path.join(BASE_DIR, "json_database", "enriched_summaries.json")



def backfill_citation_metadata(pickle_path: str, json_path: str) -> None:
    """
    Post-enrichment patch — NOT part of the original enrichment pipeline.

    Added later, after enrichment (`ChunkEnricher.enrich()`) had already
    completed and `enriched_summaries.json` was fully populated. The
    original enrichment run never captured `page_number` or `filename`
    from the chunk metadata, so this function goes back and adds them
    into the existing JSON entries by matching index position against
    the original `.pkl` chunk list (entry[i] <-> pickle_chunks[i]).

    Does NOT call the LLM. Does NOT regenerate `ai_summary`. Safe to
    run once, after confirming both files are the same length.
    """
    with open(pickle_path, 'rb') as f:
        pickle_chunks = pickle.load(f)

    with open(json_path, 'r') as f:
        enriched = json.load(f)

    if len(pickle_chunks) != len(enriched):
        print(f"Mismatch — pickle has {len(pickle_chunks)}, JSON has {len(enriched)}. Aborting.")
        return

    for i, entry in enumerate(enriched):
        chunk = pickle_chunks[i]
        entry['page_number'] = getattr(chunk.metadata, 'page_number', None)
        entry['filename'] = getattr(chunk.metadata, 'filename', None)

    with open(json_path, 'w') as f:
        json.dump(enriched, f, indent=2)

    print(f"Backfilled page_number/filename into {len(enriched)} entries.")



    if __name__ == "__main__":
        backfill_citation_metadata(CHUNKED_ELEMENTS, JSON_DB)