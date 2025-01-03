import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader

# Load API key from .env file
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_web_text(web_urls):
    text = ""
    for url in web_urls:
        # Use WebBaseLoader to load the content of a web page
        loader = WebBaseLoader(url)
        docs = loader.load()
        for doc in docs:
            text += doc.page_content  # Combine text from all loaded documents
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def contextualize_system_prompt():
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, convert_system_message_to_human=True)
    contextualize_q_chain = contextualize_q_prompt | model | StrOutputParser()
    return contextualize_q_chain

def contextualized_question(input: dict):
    if input.get("chat_history"):
        contextualize_q_chain = contextualize_system_prompt()
        return contextualize_q_chain
    else:
        return input["question"]

def get_conversational_chain():
    prompt_template = """
        You are a personal Bot assistant specializing in answering questions based on web-based context.\n
        You are provided with a set of information extracted from web pages and a user's question.\n
        Your task is to answer the user's question using only the information found in the provided context. \n
        If the question requires you to provide specific information, ensure your answer is grounded solely in the context.\n
        You must adhere to the following rules:\n
        - If you cannot find the answer to the user's question within the provided context, respond by saying you couldn't find the information and suggest rephrasing the query.\n
        - If the question pertains to a topic outside the provided context, inform the user that you don't have the necessary information.\n
        - Use bullet points or numbered lists for structured answers, but only when necessary.\n
        - Answer in the same language used by the user, either Bahasa Indonesia or English.\n
        - Do not attempt to answer any coding-related questions.\n
        - If asked for your name, respond that your name is Elena.\n
        - Avoid generating any information not explicitly provided in the context.\n\n

        Context (from web pages):\n {context}\n
        User Question: {question}\n

        Your Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, convert_system_message_to_human=True)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever | format_docs
        )
        | qa_prompt
        | model
    )    
    return rag_chain

# def response_generator_streaming(text):
#     """Generator untuk menampilkan respons secara bertahap seperti sedang mengetik."""
#     lines = text.splitlines()  # Pisahkan berdasarkan baris
#     for line in lines:
#         current_line = ""
#         for char in line:  # Tampilkan karakter demi karakter
#             current_line += char
#             yield current_line + "\n"  # Tambahkan baris baru
#             time.sleep(0.05)  # Jeda antar karakter
#         yield current_line + "\n\n"  # Tambahkan jeda antar baris

def main():
    st.set_page_config("Chat Web Pages")
    st.title("Simple chat with Web Pages")

    # Predefined list of websites
    predefined_web_urls = [
        "https://aiforindonesia.com"
    ]

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "web_processed" not in st.session_state:
        st.session_state["web_processed"] = False  # Track if processing is done

    # Process predefined websites at the start
    if not st.session_state["web_processed"]:
        with st.spinner("Processing predefined web pages..."):
            raw_text = get_web_text(predefined_web_urls)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.session_state["web_processed"] = True
        st.success("Web pages processed successfully!")

    # Display chat history
    for message in st.session_state["chat_history"]:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # Accept user input
    if prompt := st.chat_input("Say something"):
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response using the predefined web content
        rag_chain = get_conversational_chain()
        ai_msg = rag_chain.invoke(
            {"question": prompt, "chat_history": st.session_state["chat_history"]}
        )

        # Display assistant response with streaming effect
        with st.chat_message("assistant"):
            st.markdown(ai_msg.content)

        # Update chat history
        st.session_state["chat_history"].extend([HumanMessage(content=prompt), AIMessage(content=ai_msg.content)])

if __name__ == "__main__":
    main()
