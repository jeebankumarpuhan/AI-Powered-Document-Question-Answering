# scripts/rag_query.py
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Paths
INDEX_FILE = "dataset/faiss_index.bin"
CHUNKS_TEXT_FILE = "dataset/chunks_texts.json"
MODEL_NAME = "all-MiniLM-L6-v2"  # embedding model
QA_MODEL_NAME = "google/flan-t5-small"  # instruction-following QA model

def load_index():
    """Load FAISS index and chunk metadata."""
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_TEXT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def get_top_chunks(query, model, index, chunks, top_k=3):
    """Retrieve top matching chunks from FAISS for the query."""
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb.astype("float32"), top_k)
    top_results = []
    for score, idx in zip(D[0], I[0]):
        top_results.append((score, chunks[idx]["text"]))
    return top_results

def generate_answer(context, question, qa_model):
    """Generate an answer using an instruction-following model."""
    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = qa_model(prompt, max_length=100, do_sample=False)
    return response[0]["generated_text"]

def main():
    print("Loading embedding model and FAISS index...")
    embedder = SentenceTransformer(MODEL_NAME)
    index, chunks = load_index()

    print(f"Loading QA model ({QA_MODEL_NAME})...")
    qa_model = pipeline("text2text-generation", model=QA_MODEL_NAME)

    while True:
        query = input("\nEnter your question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        # Get top matching chunks
        top_chunks = get_top_chunks(query, embedder, index, chunks)
        context = "\n".join([c for _, c in top_chunks])

        # Generate answer
        answer = generate_answer(context, query, qa_model)

        print("\n🧠 AI Answer:")
        print(answer)

if __name__ == "__main__":
    main()
