from fastapi import FastAPI, UploadFile, File
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from pydantic import BaseModel
import os
from pinecone import Pinecone, ServerlessSpec

# FastAPI App Initialization
app = FastAPI()

# API Keys
API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Pinecone Initialization
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "iai-chatbot"

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Embedding dimension, adjust if needed
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

# Connect to the Pinecone index
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
vector_store = Pinecone(index=index, embedding=embeddings, text_key="text")

# Prompt Template
prompt_template = PromptTemplate(
    template="""\
    Kamu adalah asisten chatbot yang membantu mengenalkan perusahaan Indonesia AI.
    Jawab pertanyaan berdasarkan konteks berikut. Jika kamu tidak tahu jawab, katakan "maaf saya kurang tahu."
    Sapa klien dengan ramah.\n\nKonteks:\n{context}\n\nPertanyaan: {question}\nJawaban: """,
    input_variables=["context", "question"],
)

chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Helper Functions
def get_pdf_text(file):
    """Extract text from uploaded PDF."""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise ValueError(f"Gagal membaca file PDF: {str(e)}")

def get_text_chunks(text):
    """Split text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Endpoint Models
class ChatInput(BaseModel):
    question: str
    chat_history: list = []

# API Endpoints
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file, storing embeddings in Pinecone.
    """
    try:
        # Extract text and split into chunks
        text = get_pdf_text(file.file)
        text_chunks = get_text_chunks(text)

        # Add text chunks to Pinecone
        vector_store.add_texts(texts=text_chunks)
        return {"message": "File berhasil diproses dan disimpan di Pinecone"}
    except Exception as e:
        return {"error": f"Terjadi kesalahan saat memproses file: {str(e)}"}

@app.post("/chat")
async def chat(input: ChatInput):
    """
    Handle chat requests by retrieving relevant context and generating responses.
    """
    try:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # Create conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt_template},
        )

        # Generate answer
        output = qa_chain({"question": input.question, "chat_history": input.chat_history})
        response = output["answer"]

        # Optionally include sources
        sources = [
            {"source": doc.metadata.get("source", "Unknown"), "content": doc.page_content}
            for doc in output.get("source_documents", [])
        ]

        return {"answer": response, "sources": sources}
    except Exception as e:
        return {"error": f"Terjadi kesalahan saat memproses permintaan: {str(e)}"}
