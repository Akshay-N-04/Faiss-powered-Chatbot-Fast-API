from fastapi import APIRouter
from train import train_model

train_router = APIRouter()

@train_router.get("/")
def train_chatbot():
    train_model()
    return {"message": "Model training completed successfully"}
