import sqlite3

DB_PATH = "data/qa_database.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL UNIQUE, 
            answer TEXT NOT NULL UNIQUE,
            faiss_id INTEGER DEFAULT NULL
        )
    """)
    conn.commit()
    conn.close()

def connect_db():
    return sqlite3.connect(DB_PATH)

def fetch_qa_pairs():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer, faiss_id FROM qa_pairs order by id asc")   
    data = cursor.fetchall()
    conn.close()
    return data

def add_qa_pair(question, answer):
    """Adds a new Q&A pair to the database only if it's unique."""
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM qa_pairs WHERE question = ?", (question,))
    if cursor.fetchone():
        conn.close()
        return {"error": "Duplicate entry! This question already exists in the database."}

    cursor.execute("INSERT INTO qa_pairs (question, answer, faiss_id) VALUES (?, ?, NULL)", (question, answer,))
    conn.commit()
    conn.close()

def delete_question_from_db(question: str):
    """Deletes a Q&A pair from the database and returns True if deleted."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM qa_pairs WHERE question = ?", (question,))
    result = cursor.fetchone()
    if not result:
        conn.close()
        return False  # Question not found

    cursor.execute("DELETE FROM qa_pairs WHERE question = ?", (question,))
    conn.commit()
    conn.close()
    return True


init_db()
