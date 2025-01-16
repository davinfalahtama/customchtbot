#rag.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from pinecone import Pinecone as PineconeClient
import os
from history import ChatHistory

class RAGPipeline:
    def __init__(self, openai_api_key, pinecone_api_key, pinecone_env, dataset_path, index_name, user_id):
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_env = pinecone_env
        self.dataset_path = dataset_path
        self.index_name = index_name
        self.qa_chain = self._initialize_resources()
        self.chat_history = ChatHistory(user_id)

    def _initialize_resources(self):
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            pinecone_client = PineconeClient(api_key=self.pinecone_api_key)
            index = pinecone_client.Index(self.index_name)
            vector_store = Pinecone(index, embeddings, "text")

            with open(self.dataset_path, "rb") as f:
                pdf_reader = PdfReader(f)
                raw_text = "".join(page.extract_text() for page in pdf_reader.pages)
            text_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(raw_text)

            if vector_store._index.describe_index_stats().total_vector_count == 0:
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
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=chat_model,
                retriever=retriever,
                combine_docs_chain_kwargs={"prompt": prompt_template}, # return_source_documents dihapus
            )

            print("RAG resources initialized successfully.")
            return qa_chain

        except Exception as e:
            print(f"Error during RAG initialization: {e}")
            raise

    def query(self, question):
        full_history = self.chat_history.get_history()
        chat_history_list = [(msg["type"], msg["content"]) for msg in full_history[-2:]]
        output = self.qa_chain({"question": question, "chat_history": chat_history_list})
        self.chat_history.add_message("user", question)
        self.chat_history.add_message("bot", output['answer'])
        return output['answer']

    def clear_history(self):
        self.chat_history.clear_history()

    def delete_message(self, message_id):
        self.chat_history.delete_message(message_id)