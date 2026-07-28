import os
import json
 
# Tunable detection thresholds
MAX_HEADER_WORDS = 45   # a chunk this short or shorter is a header candidate
                         # (see fix_orphaned_headers.md for how this was chosen)
 
 
def word_count(text: str) -> int:
    return len(text.split())

 
def is_header_candidate(chunk: dict) -> bool:
    """A chunk is a header candidate if it's classified as pure text and is short."""
    types = chunk.get("metadata", {}).get("types_found", ["text"])
    content = chunk.get("page_content", "")
    return types == ["text"] and word_count(content) <= MAX_HEADER_WORDS
 
 
def is_table_chunk(chunk: dict) -> bool:
    types = chunk.get("metadata", {}).get("types_found", ["text"])
    return "table" in types
 
 
def merge_orphaned_headers(data: dict) -> tuple[dict, list[dict]]:
    """
    Returns:
        fixed_data: dict of chunks with orphaned headers merged into their
                    following table chunk, using the same key scheme as input
        merge_log:  list of records describing every merge that was performed,
                    for auditing / before-after inspection
    """
    keys = sorted(data.keys(), key=lambda k: int(k))
 
    merged_into_next = set()   # keys of header chunks that got merged away
    merge_log = []
 
    for i in range(len(keys) - 1):
        k, k_next = keys[i], keys[i + 1]
 
        if k in merged_into_next:
            continue  # already consumed by a previous merge
 
        chunk = data[k]
        next_chunk = data[k_next]
 
        if is_header_candidate(chunk) and is_table_chunk(next_chunk):
            header_text = chunk["page_content"].strip()
            original_table_text = next_chunk["page_content"]
 
            # Prepend header text so the table chunk carries its own identity
            next_chunk["page_content"] = f"{header_text}\n\n{original_table_text}"
 
            merged_into_next.add(k)
            merge_log.append({
                "header_key": k,
                "table_key": k_next,
                "header_text": header_text,
                "table_preview": original_table_text[:120],
            })
 
    fixed_data = {k: v for k, v in data.items() if k not in merged_into_next}
 
    return fixed_data, merge_log
 



def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_dir = os.path.join(base_dir, "json_database")
    os.makedirs(json_dir, exist_ok=True)

    input_path = os.path.join(json_dir, "final_processed_chunks.json")
    fixed_output_path = os.path.join(json_dir, "final_processed_chunks_FIXED.json")
    log_output_path = os.path.join(json_dir, "merge_log.json")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_count = len(data)

    fixed_data, merge_log = merge_orphaned_headers(data)

    with open(fixed_output_path, "w", encoding="utf-8") as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)

    with open(log_output_path, "w", encoding="utf-8") as f:
        json.dump(merge_log, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()