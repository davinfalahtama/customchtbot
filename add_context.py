import os
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load PDF file
loader = PyPDFLoader("dataset/indonesia-ai-dataset.pdf")
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(documents)

# Initialize embeddings and upload to Pinecone
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
index_name = "chatbot-iai"

# Upload to Pinecone
pinecone = PineconeVectorStore.from_documents(
    split_docs, 
    embeddings, 
    index_name=index_name
)