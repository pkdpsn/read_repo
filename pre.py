import pandas as pd
import json
import hashlib
import os
from tqdm import tqdm

input_file = "/kaggle/input/.../chunk_ranking_kaggle_eval.jsonl"

chunks_file = "chunks.jsonl"
qa_file = "qa_with_qrel.jsonl"

# Track already written hashes (avoid duplicates across runs too)
seen_hashes = set()

# OPTIONAL: load existing hashes if rerunning
if os.path.exists(chunks_file):
    with open(chunks_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            seen_hashes.add(obj["chunk_id"])


def get_chunk_id(text):
    # stable hash
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    chunk_id = f"c_{h}"
    
    # write only if new
    if chunk_id not in seen_hashes:
        with open(chunks_file, "a") as f:
            f.write(json.dumps({
                "chunk_id": chunk_id,
                "text": text
            }) + "\n")
        seen_hashes.add(chunk_id)
    
    return chunk_id


# STREAM processing
for df in pd.read_json(input_file, lines=True, chunksize=10):
    
    for _, row in df.iterrows():
        
        qid = row["uuid"]
        messages = row["messages"]
        qrel_raw = row["qrel"]
        
        # Extract question
        question = messages[0]["content"]
        
        # 🔴 IMPORTANT: adjust this based on actual structure
        chunk_texts = []
        for m in messages:
            if m["role"] == "assistant":
                chunk_texts.append(m["content"])
        
        # Map chunks → IDs
        chunk_ids = [get_chunk_id(text) for text in chunk_texts]
        
        # Map qrel (index → chunk_id)
        qrel_mapped = {}
        for idx, score in qrel_raw.items():
            idx = int(idx)
            if idx < len(chunk_ids):
                qrel_mapped[chunk_ids[idx]] = score
        
        # Save QA
        with open(qa_file, "a") as f:
            f.write(json.dumps({
                "qid": qid,
                "question": question,
                "chunks": chunk_ids,
                "qrel": qrel_mapped
            }) + "\n")
