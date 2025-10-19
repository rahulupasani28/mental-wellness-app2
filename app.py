import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

st.title("Mental Wellness Chatbot")

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    messages = st.session_state.store[session_id].messages[-10:]  
    st.session_state.store[session_id].messages = messages
    return st.session_state.store[session_id]

model = ChatGroq(
    groq_api_key="YOUR_GROQ_KEY",
    model_name="llama-3.3-70b-versatile",
    temperature=0.6
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a supportive mental wellness chatbot."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

chain = prompt | model | StrOutputParser()

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

user_input = st.text_input("How can I help you today?")

if user_input:
    response = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "user-session"}}
    )
    st.write(response)
