from fastapi import APIRouter, HTTPException
from models.model_handler import retrieve_answer

chat_router = APIRouter()

@chat_router.get("/")
def chatbot_response(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    response, similarity_score = retrieve_answer(query)
    
    return {"query": query, "response": response, "similarity": similarity_score}
