# scripts/extract_and_chunk.py
import os
from PyPDF2 import PdfReader
import json
import re

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder containing raw docs
DATA_DIR = os.path.join(BASE_DIR, "dataset", "raw_docs")

# Output file for chunks
OUT_DIR = os.path.join(BASE_DIR, "dataset")
OUT_CHUNKS = os.path.join(OUT_DIR, "chunks.jsonl")

CHUNK_SIZE = 800   # chars
OVERLAP = 200      # chars

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def clean_text(t):
    # simple clean
    return re.sub(r"\s+", " ", t).strip()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = max(end - overlap, end)
    return chunks

def main():
    # check if DATA_DIR exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: DATA_DIR does not exist: {DATA_DIR}")
        return

    # create output directory if not exists
    os.makedirs(OUT_DIR, exist_ok=True)

    all_chunks = []

    for fname in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, fname)
        if fname.lower().endswith(".pdf"):
            text = extract_text_from_pdf(path)
        elif fname.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            continue
        text = clean_text(text)
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            all_chunks.append({
                "source": fname,
                "chunk_id": f"{fname}__{i}",
                "text": c
            })

    # save as jsonl
    with open(OUT_CHUNKS, "w", encoding="utf-8") as out:
        for chunk in all_chunks:
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_chunks)} chunks to {OUT_CHUNKS}")

if __name__ == "__main__":
    main()
