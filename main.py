from fastapi import FastAPI, HTTPException, Query, Response, Request, Path
from rag import RAGPipeline
import os
from dotenv import load_dotenv
import uuid
from history import ChatHistory

load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
DATASET_PATH = os.getenv("DATASET_PATH", "dataset/indonesia-ai-dataset.pdf")
INDEX_NAME = os.getenv("INDEX_NAME", "iai-chatbot")

try:
    # Inisialisasi awal dengan user_id None
    rag_pipeline = RAGPipeline(OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, DATASET_PATH, INDEX_NAME, None)
except Exception as e:
    print(f"Failed to initialize RAG pipeline: {e}")
    exit(1)

@app.post("/chat")
async def chat_endpoint(request: Request, response: Response, text: str = Query(..., description="Teks pertanyaan")):
    try:
        user_id = request.cookies.get("user_id")

        if not user_id:
            user_id = str(uuid.uuid4())
            response.set_cookie(key="user_id", value=user_id, httponly=True)
        rag_pipeline.chat_history = ChatHistory(user_id)
        answer = rag_pipeline.query(text)
        return {"answer": answer, "user_id": user_id}
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_history")
async def clear_history_endpoint(request: Request):
    try:
        user_id = request.cookies.get("user_id")
        if user_id:
            rag_pipeline.chat_history = ChatHistory(user_id)
            rag_pipeline.clear_history()
            return {"message": "Chat history cleared", "user_id": user_id}
        else:
            return {"message": "No user ID found, nothing to clear."}
    except Exception as e:
        print(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/message/{message_id}")
async def delete_message_endpoint(request: Request, message_id: str = Path(..., description="ID pesan yang akan dihapus")):
    try:
        user_id = request.cookies.get("user_id")
        if user_id:
            rag_pipeline.chat_history = ChatHistory(user_id)
            rag_pipeline.delete_message(message_id)
            return {"message": f"Message with ID {message_id} deleted", "user_id": user_id}
        else:
            return {"message": "No user ID found, cannot delete message."}

    except Exception as e:
        print(f"Error deleting message: {e}")
        raise HTTPException(status_code=500, detail=str(e))