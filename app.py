import os
import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Mental Wellness Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS ...
st.markdown("""<style>
/* your custom CSS kept exactly as is */
</style>""", unsafe_allow_html=True)

# Title / Disclaimer
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

# Sidebar
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

# Load API Key
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables.")
    st.stop()

# System prompt
system_prompt = """
You are a compassionate and supportive mental wellness assistant.
Offer practical coping strategies, mindfulness exercises, or general advice.
Remind users you are not a licensed therapist.
Keep responses positive and empowering.
"""

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

# Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Prompt Template (includes memory)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Build Runnable Chain manually (modern LangChain replacement for LLMChain)
chain = (
    prompt_template
    | llm
    | StrOutputParser()
)

# Header
st.markdown('<div class="app-header">🧠 Mental Wellness Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Emergency Table
emergency_helplines = pd.DataFrame({
    "Country": ["USA", "India", "UK", "Canada", "Australia", "South Africa", "Japan"],
    "Emergency Helpline": ["911", "112", "999", "911", "000", "10111", "110"]
})
st.markdown("### Emergency Helplines Worldwide")
st.markdown("Below are some emergency helplines...")
st.dataframe(emergency_helplines, use_container_width=True)

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_input = st.chat_input("How are you feeling today?")

if "user_input" in st.session_state:
    user_input = st.session_state.user_input
    del st.session_state.user_input

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({"input": user_input})
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown('</div>', unsafe_allow_html=True)
