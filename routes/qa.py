from fastapi import APIRouter, HTTPException
from database.db_handler import add_qa_pair
from database.db_handler import connect_db
from models.model_handler import remove_from_faiss

qa_router = APIRouter()
@qa_router.post("/add_qa/")
def add_qa(question: str, answer: str):
    """Adds a new Q&A pair to the database and updates FAISS."""
    if not question.strip() or not answer.strip():
        raise HTTPException(status_code=400, detail="Question and Answer cannot be empty.")

    add_qa_pair(question.lower(), answer.lower())  # Add to database
    
    return {"message": "Q&A added successfully!"}

@qa_router.delete("/delete/")
def delete_qa(question: str):
    conn = connect_db()
    cursor = conn.cursor()

    # Check if the question exists
    cursor.execute("SELECT id, faiss_id FROM qa_pairs WHERE question = ?", (question,))
    result = cursor.fetchone()

    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="Question not found in the database")

    question_id, faiss_id = result

    # Delete from database
    cursor.execute("DELETE FROM qa_pairs WHERE id = ?", (question_id,))
    conn.commit()
    conn.close()

    # Remove from FAISS if a valid faiss_id exists
    if faiss_id is not None:
        success = remove_from_faiss(faiss_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update FAISS index")

    return {"message": "Question deleted successfully from both DB and FAISS"}