import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Set up Streamlit page configuration for a light, modern look
st.set_page_config(
    page_title="Mental Wellness Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern light theme
st.markdown("""
    <style>
    body { background-color: #f0f4f8; color: #333333; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stApp > header { background-color: #ffffff; }
    .css-1lcbmhc { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    .st-chat-message { border-radius: 12px; padding: 12px; margin-bottom: 12px; }
    .st-chat-message.user { background-color: #e3f2fd; color: #1565c0; }
    .st-chat-message.assistant { background-color: #ffffff; color: #333333; border: 1px solid #e0e0e0; }
    .stButton > button { background-color: #4caf50; color: white; border: none; border-radius: 8px; padding: 8px 16px; }
    .stButton > button:hover { background-color: #388e3c; }
    .st-info { background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 12px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# Mental health disclaimer
st.title("🧠 Mental Wellness Chatbot")
st.info("""
**Important Note on Mental Health:**  
This chatbot provides general support and coping strategies. It is not a substitute for professional advice.  
Seek help from a qualified professional if needed.  
Resources:  
- National Suicide Prevention Lifeline (US): 988  
- Crisis Text Line: Text HOME to 741741  
- International help: https://www.befrienders.org
""")

# Sidebar with predefined prompts
st.sidebar.title("Quick Start Prompts")
predefined_prompts = [
    "I'm feeling anxious about work. What can I do?",
    "How can I practice mindfulness daily?",
    "I'm having trouble sleeping. Any tips?",
    "What are some ways to build self-esteem?",
    "I feel overwhelmed. Help me prioritize."
]

for prompt in predefined_prompts:
    if st.sidebar.button(prompt):
        st.session_state.user_input = prompt

# Load API key from environment variable (from GitHub Secrets)
groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please set it as a GitHub Secret or deployment environment variable.")
    st.stop()

# System prompt
system_prompt = """
You are a compassionate and supportive mental wellness assistant. Offer practical coping strategies, mindfulness exercises, or general advice. Remind users you are not a licensed therapist. Keep responses positive and empowering.
"""

# LangChain setup
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

memory = ConversationBufferMemory(memory_key="chat_history")
chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("How are you feeling today?")

if "user_input" in st.session_state:
    user_input = st.session_state.user_input
    del st.session_state.user_input

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.run(input=user_input)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
