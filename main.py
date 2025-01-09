from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
import time
from langchain.prompts import PromptTemplate
from pinecone import Pinecone as PineconeClient
import os

app = FastAPI()

# Configuration (move to environment variables or config file for production)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
DATASET_PATH = "dataset/indonesia-ai-dataset.pdf"  # Make sure this path is correct in your deployment
INDEX_NAME = "iai-chatbot"

# Initialize resources outside of request handlers for efficiency
try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone_client = PineconeClient(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
        )
    index = pinecone_client.Index(INDEX_NAME)
    vector_store = Pinecone(index, embeddings, "text")

    # Load and process the PDF only once at startup
    with open(DATASET_PATH, "rb") as f: # open in binary mode
        pdf_reader = PdfReader(f)
        raw_text = "".join(page.extract_text() for page in pdf_reader.pages)
    text_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(raw_text)
    if vector_store._index.describe_index_stats().total_vector_count == 0: # only add if the index is empty
        vector_store.add_texts(texts=text_chunks)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    prompt_template = PromptTemplate(
        template=("""kamu adalah seorang asisten chatbot yang akan membantu client mengenal perusahaan Indonesia AI lebih detil
        seperti mengenalkan perusahaan, layanan atau program yang ditawarkan dan informasi lainnya.
        jawab pertanyaan dari konteks yang diberikan, kalau kamu tidak tahu bilang saja "maaf saya kurang tau."
        usahakan sapa dan jawab pertanyaan client dengan seramah mungkin ya.\n\n"
        "Konteks:\n{context}\n\nPertanyaan: {question}\nJawaban: """),
        input_variables=["context", "question"],
    )
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    # create the chain outside of the request
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt_template},
    )

    print("Resources initialized successfully.")

except Exception as e:
    print(f"Error during initialization: {e}")
    raise  # Re-raise the exception to prevent the app from starting

class ChatRequest(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = []

class ChatResponse(BaseModel):
    answer: str
    source_documents: List[str] = []

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        output = qa_chain({"question": request.question, "chat_history": request.chat_history})
        response = output["answer"]
        sources = [doc.page_content for doc in output.get("source_documents", [])]

        return ChatResponse(answer=response, source_documents=())
    except Exception as e:
        print(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}