import sqlite3
from llama_cpp import Llama
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Load Sentence Transformer for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Path to Llama 2 GGUF model
MODEL_PATH = "./llama-2-7b-chat.Q4_K_M.gguf"

# Load Llama 2 model
llm = Llama(model_path=MODEL_PATH)

# Load FAISS index
def load_faiss_index():
    """Loads FAISS index from file."""
    try:
        index = faiss.read_index("data/faiss_index.bin")
        print(" Loaded FAISS index from file!")
        return index
    except Exception:
        print(" FAISS index not found!")
        return None

faiss_index = load_faiss_index()

def reload_faiss():
    global faiss_index
    faiss_index = load_faiss_index()
    print("FAISS index reloaded successfully!")

def retrieve_answer(query, threshold=0.6):
    """Retrieve best-matching answer using FAISS and generate response with Llama 2."""

    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # Normalize embedding for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Search FAISS
    distances, idx = faiss_index.search(query_embedding, 1)
    best_match_index = idx[0][0]
    best_match_distance = distances[0][0]

    similarity_score = (best_match_distance + 1) / 2

    if similarity_score < threshold or best_match_index == -1:
        return "I don't know. Can you provide more details?[1]", similarity_score

    # Fetch the answer using FAISS ID from SQLite
    conn = sqlite3.connect("data/qa_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT answer FROM qa_pairs WHERE faiss_id = ?", (int(best_match_index),))
    result = cursor.fetchone()
    conn.close()

    if not result:
        return "I don't know. Can you provide more details?[2]", similarity_score

    stored_answer = result[0]
    prompt = f"""
    Context: {stored_answer}
    Query: {query}
    Short Answer: """
    
    response = llm(prompt, max_tokens=200)  # Generate response using Llama 2
    final_response = response["choices"][0]["text"].strip()

    return final_response, similarity_score

def remove_from_faiss(faiss_id):
    """Removes an entry from the FAISS index using remove_ids() and updates the index file."""
    global faiss_index
    if faiss_index is None:
        raise RuntimeError("FAISS index not loaded!")
    
    # Creating an ID selector for the given faiss_id
    try:
        id_selector = faiss.IDSelectorArray(np.array([faiss_id], dtype=np.int64))
        faiss_index.remove_ids(id_selector)
        faiss.write_index(faiss_index, "data/faiss_index.bin")
        print(f"Successfully removed FAISS ID {faiss_id} and updated the FAISS index.")
        return True
    except Exception as e:
        print(f" Failed to remove FAISS ID {faiss_id}: {str(e)}")
        return False
