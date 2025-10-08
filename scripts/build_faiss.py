# scripts/build_faiss.py
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "dataset/chunks.jsonl"
INDEX_FILE = "dataset/faiss_index.bin"
CHUNKS_TEXT_FILE = "dataset/chunks_texts.json"


MODEL_NAME = "all-MiniLM-L6-v2"   # local model

def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    model = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    print("Encoding", len(texts), "chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # use inner product (cosine after normalization)
    # Normalize for cosine similarity if using IP
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, INDEX_FILE)
    # save metadata (source, chunk_id, text)
    with open(CHUNKS_TEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print("Index saved:", INDEX_FILE)

if __name__ == "__main__":
    main()
