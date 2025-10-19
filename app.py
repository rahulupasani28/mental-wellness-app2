import os
import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Set up Streamlit page configuration for a modern look
st.set_page_config(
    page_title="Mental Wellness Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to align the app name/logo on the left side and make it stick to the top
st.markdown("""
    <style>
    /* Make header sticky at the top */
    .stApp > header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        display: flex;
        justify-content: flex-start;
        align-items: center;
        padding: 10px 20px;
        background-color: #ffffff;
        border-bottom: 2px solid #e0e0e0;
        z-index: 1000; /* Ensure the header is on top */
    }
    .stApp > header .css-1d391kg {
        display: none;
    }
    .app-header {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
        margin-left: 10px;
    }
    /* Adjust the layout to accommodate the sticky header */
    .main-content {
        margin-top: 100px; /* Ensure content starts below sticky header */
    }
    .css-1lcbmhc {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    .st-chat-message { 
        border-radius: 12px; 
        padding: 12px; 
        margin-bottom: 12px; 
    }
    .st-chat-message.user { 
        background-color: #e3f2fd; 
        color: #1565c0; 
    }
    .st-chat-message.assistant { 
        background-color: #ffffff; 
        color: #333333; 
        border: 1px solid #e0e0e0; 
    }
    .stButton > button { 
        background-color: #4caf50; 
        color: white; 
        border: none; 
        border-radius: 8px; 
        padding: 8px 16px; 
    }
    .stButton > button:hover { 
        background-color: #388e3c; 
    }
    .st-info { 
        background-color: #e8f5e9; 
        border-left: 5px solid #4caf50; 
        padding: 12px; 
        border-radius: 8px;
        font-size: 14px;
    }
    .stMarkdown {
        font-size: 16px;
        line-height: 1.6;
    }
    .table-container {
        margin-top: 30px;
        margin-bottom: 30px;
    }
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
- International help: [Befrienders Worldwide](https://www.befrienders.org)
""")

# Sidebar with predefined prompts
st.sidebar.title("Quick Start Prompts")
predefined_prompts = [
    "I'm feeling anxious about work. What can I do?",
    "How can I practice mindfulness daily?",
    "Begin the anxiety test and ask me the questions.",
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

# Add app name/logo to the header
st.markdown('<div class="app-header">🧠 Mental Wellness Chatbot</div>', unsafe_allow_html=True)

# Main content (with margin for sticky header)
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Display a table for national emergency helplines
emergency_helplines = pd.DataFrame({
    "Country": ["USA", "India", "UK", "Canada", "Australia", "South Africa", "Japan"],
    "Emergency Helpline": ["911", "112", "999", "911", "000", "10111", "110"]
})

# Display the table under the main content
st.markdown("### Emergency Helplines Worldwide")
st.markdown("""
    Below are some emergency helplines for countries around the world. In case of an emergency, please don't hesitate to reach out to the appropriate number.
""")

st.markdown("""
    <div class="table-container">
    """, unsafe_allow_html=True)

st.dataframe(emergency_helplines, use_container_width=True)

st.markdown("""
    </div>
""", unsafe_allow_html=True)

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

st.markdown('</div>', unsafe_allow_html=True)


