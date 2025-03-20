import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from database.db_handler import fetch_qa_pairs
from models.model_handler import reload_faiss

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = embedding_model.get_sentence_embedding_dimension()  # Should be 384 for MiniLM

FAISS_INDEX_PATH = "data/faiss_index.bin"

def train_model():
    """
    Rebuilds the entire FAISS index from all Q&A pairs in the database
    and updates each record's faiss_id.
    """
    # Connect to database
    conn = sqlite3.connect("data/qa_database.db")
    cursor = conn.cursor()

    # Fetch all Q&A pairs (all records)
    cursor.execute("SELECT id, question FROM qa_pairs ORDER BY id ASC")
    all_data = cursor.fetchall()

    if not all_data:
        print(" No Q&A data found in the database!")
        # Create and save an empty FAISS index
        index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(" Empty FAISS index created successfully!")
        conn.close()
        return "No data to train! Please add Q&A pairs."

    # Separate IDs and questions
    record_ids, questions = zip(*all_data)

    # Compute embeddings for all questions
    question_embeddings = embedding_model.encode(questions)
    # Normalize embeddings for cosine similarity
    question_embeddings = question_embeddings / np.linalg.norm(question_embeddings, axis=1, keepdims=True)

    # Create a new FAISS index with IndexIDMap to support add_with_ids()
    index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
    # Create new FAISS IDs for all records (e.g. from 0 to N-1)
    new_ids = np.array(range(len(record_ids)), dtype=np.int64)
    index.add_with_ids(np.array(question_embeddings, dtype=np.float32), new_ids)

    # Update database: assign new FAISS IDs to each record.
    for i, rec_id in enumerate(record_ids):
        cursor.execute("UPDATE qa_pairs SET faiss_id = ? WHERE id = ?", (int(new_ids[i]), rec_id))

    conn.commit()
    conn.close()

    # Save the updated FAISS index and reload it in memory
    faiss.write_index(index, FAISS_INDEX_PATH)
    reload_faiss()

    print(f" FAISS trained with {len(questions)} Q&A pairs!")
    return "Model trained successfully!"

if __name__ == "__main__":
    train_model()
