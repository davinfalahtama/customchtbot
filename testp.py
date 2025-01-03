import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import time
from langchain.prompts import PromptTemplate
import pinecone

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

def get_pdf_text(pdf_paths):
    text = ""
    for file_path in pdf_paths:
        pdf_reader = PdfReader(file_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Pinecone index setup
    index_name = "iai-chatbot"
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=1536,  # Adjust according to embedding model size
            metric="cosine",
        )

    # Connect to the existing index
    index = pinecone.Index(index_name)

    # Initialize vector store using LangChain Pinecone
    vector_store = Pinecone(
        index=index,
        embedding=embeddings,
        text_key="text",
    )

    # Add texts to vector store
    vector_store.add_texts(texts=text_chunks)
    return vector_store

def format_chat_history(chat_history):
    """Format chat history into a list of tuples (role, content)."""
    return [(message["role"], message["content"]) for message in chat_history]

def main():
    st.set_page_config(
        page_title="Chat Documents",
        page_icon="📄",
        initial_sidebar_state="expanded",
    )
    st.title("IAI Chatbot (Pinecone)")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    dataset_path = "dataset/indonesia-ai-dataset.pdf"
    pdf_docs = [dataset_path]

    with st.spinner("Memproses dokumen..."):
        start_time = time.time()
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        vector_store = get_vector_store(text_chunks)
        end_time = time.time()
        st.success(f"Dokumen berhasil diproses dalam {end_time - start_time:.2f} detik.")

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Prompt template
    prompt_template = PromptTemplate(
        template=("""kamu adalah seorang asisten chatbot yang akan membantu client mengenal perusahaan Indonesia AI lebih detil
        seperti mengenalkan perusahaan, layanan atau program yang ditawarkan dan informasi lainnya.
        jawab pertanyaan dari konteks yang diberikan, kalau kamu tidak tahu bilang saja "maaf saya kurang tau."
        usahakan sapa dan jawab pertanyaan client dengan seramah mungkin ya.\n\n"
        "Konteks:\n{context}\n\nPertanyaan: {question}\nJawaban: """),
        input_variables=["context", "question"],
    )

    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt_template},
    )

    for message in st.session_state.get("chat_history", []):
        with st.chat_message("user" if message["role"] == "user" else "assistant"):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tanyakan sesuatu..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["chat_history"].append({"role": "user", "content": prompt})

        with st.spinner("Mencari jawaban..."):
            formatted_history = format_chat_history(st.session_state["chat_history"])
            
            # Get all outputs using __call__
            output = qa_chain({"question": prompt, "chat_history": formatted_history})
            
            # Extract answer
            response = output["answer"]

            # (Optional) Show source documents
            sources = output.get("source_documents", [])
        
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state["chat_history"].append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
