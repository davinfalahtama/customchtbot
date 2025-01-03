import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough

# init important varible
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL = 'gpt-4o-mini'
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

parser = StrOutputParser()

template = """
kamu adalah seorang asisten chatbot yang akan membantu client mengenal perusahaan Indonesia AI lebih detil
seperti mengenalkan perusahaan, layanan atau program yang ditawarkan dan informasi lainnya.
jawab pertanyaan dari konteks yang diberikan, kalau kamu tidak tahu bilang saja "maaf saya kurang tau."
usahakan sapa dan jawab pertanyaan client dengan seramah mungkin ya
konteks: {context}
pertanyaan: {question}
"""

prompt = PromptTemplate.from_template(template)

# connect into vector database
pinecone = PineconeVectorStore(
    embedding=embeddings,
    index_name='iai-chatbot',
)

# create data pipeline
chain = (
    {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

# Set up the app layout
st.set_page_config(page_title="Custom Chatbot", layout="wide")

# Set default middleware
middleware = False
try:
    if st.query_params.access == "indo-ai":
        middleware = True
except Exception as e:
    middleware = False

if middleware:
    # Initialize session state to store chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def send_message():
        user_message = st.session_state.user_input
        if user_message:
            st.session_state.messages.append(
                {"role": "user", "text": user_message})
            # Placeholder for AI response - replace with actual AI call
            ai_response = f"{chain.invoke(user_message)}"
            st.session_state.messages.append(
                {"role": "ai", "text": ai_response})
            st.session_state.user_input = ""

    # Main chat display
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                        <div style="background-color: #dcf8c6; padding: 10px; border-radius: 10px; max-width: 80%; color:black;">
                            You: {message['text']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                        <div style="background-color: #f1f0f0; padding: 10px; border-radius: 10px; max-width: 80%; color:black;">
                            Custom Chatbot: {message['text']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Chat input at the bottom
    st.text_input(
        "Enter your message:",
        key="user_input",
        on_change=send_message
    )

    # A bit of styling to make the chat look better
    st.markdown(
        """
        <style>
        .css-1d391kg {max-width: 800px; margin: auto;}
        .css-1d391kg .markdown-text-container {max-width: 800px; margin: auto;}
        .stTextInput {position: fixed; bottom: 10px; width: 70%; left: 50%; transform: translateX(-50%);}
        .stTextInput input {width: 100%;} /* Ensures the input field takes the full width */
        .stContainer {max-width: 800px; margin-left: auto; margin-right: auto;} /* Center the chat container */
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div style="display:flex;align-items:center;justify-content:center;height:90vh;width:100%">
            <h2>Sorry, you don't have access</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
