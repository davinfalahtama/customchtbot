import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
import time
from langchain.prompts import PromptTemplate

# Access the API key from Streamlit secrets
API_KEY = st.secrets["OPENAI_API_KEY"]

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
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
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
    st.title("IAI Chatbot (FAISS)")

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
        template=("""\
                kamu adalah seorang asisten chatbot yang akan membantu client mengenal perusahaan Indonesia AI lebih detil
                seperti mengenalkan perusahaan, layanan atau program yang ditawarkan dan informasi lainnya.
                jawab pertanyaan dari konteks yang diberikan, kalau kamu tidak tahu bilang saja "maaf saya kurang tau."
                usahakan sapa dan jawab pertanyaan client dengan seramah mungkin ya.\n\n"
                "Konteks:\n{context}\n\nPertanyaan: {question}\nJawaban: """
        ),
        input_variables=["context", "question"],
    )

    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # Gunakan model GPT-4
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
            
            # Gunakan __call__ untuk mendapatkan semua output
            output = qa_chain({"question": prompt, "chat_history": formatted_history})
            
            # Ambil hanya jawaban
            response = output["answer"]

            # (Opsional) Ambil sumber dokumen jika ingin ditampilkan
            sources = output.get("source_documents", [])
        
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state["chat_history"].append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
