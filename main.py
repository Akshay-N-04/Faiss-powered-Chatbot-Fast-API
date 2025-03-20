from fastapi import FastAPI
from routes.chat import chat_router
from routes.train import train_router
from routes.qa import qa_router
import gradio as gr
from database.db_handler import add_qa_pair
from models.model_handler import retrieve_answer, reload_faiss
from train import train_model

app = FastAPI(title="Chatbot API with FastAPI + Gradio + Llama 2")

app.include_router(chat_router, prefix="/chat")
app.include_router(train_router, prefix="/train")
app.include_router(qa_router, prefix="/qa")

@app.get("/") 
def root():
    return {"message": "Chatbot API is running with Llama 2!"}

# Add Q&A
def add_question_answer(question, answer):
    add_qa_pair(question, answer)
    return "Q&A pair added successfully!"

# Chatbot function
def chat_with_model(user_input):
    response = retrieve_answer(user_input)[0]
    return response

# Train Model
def retrain_model():
    status = train_model()  # Train FAISS
    reload_faiss()  #  Reload FAISS after training
    return status 

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Llama 2 Chatbot")
    
    with gr.Row():
        user_query = gr.Textbox(label="Enter your question:")
        chat_button = gr.Button("Chat")
    
    chatbot_output = gr.Textbox(label="Chatbot Response", interactive=False)
    chat_button.click(chat_with_model, inputs=user_query, outputs=chatbot_output)
    
    gr.Markdown("## Add Q&A to Database")
    with gr.Row():
        question_input = gr.Textbox(label="Question")
        answer_input = gr.Textbox(label="Answer")
        add_button = gr.Button("Add Q&A")
    
    status_output = gr.Textbox(label="Status", interactive=False)
    add_button.click(add_question_answer, inputs=[question_input, answer_input], outputs=status_output)
    
    gr.Markdown("## Train Model")
    train_button = gr.Button("Train")
    train_status = gr.Textbox(label="Training Status", interactive=False)
    
    train_button.click(retrain_model, outputs=train_status)

# Launch Gradio App
if __name__ == "__main__":
    demo.launch()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
