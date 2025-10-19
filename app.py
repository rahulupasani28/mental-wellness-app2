import os
import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

st.set_page_config(
    page_title="Mental Wellness Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp > header {position: fixed; top: 0; left: 0; width: 100%; display: flex; justify-content: flex-start; align-items: center; padding: 10px 20px; background-color: #ffffff; border-bottom: 2px solid #e0e0e0; z-index: 1000;}
.stApp > header .css-1d391kg { display: none; }
.app-header { font-size: 24px; font-weight: bold; color: #4CAF50; margin-left: 10px; }
.main-content { margin-top: 100px; }
.st-chat-message { border-radius: 12px; padding: 12px; margin-bottom: 12px; }
.st-chat-message.user { background-color: #e3f2fd; color: #1565c0; }
.st-chat-message.assistant { background-color: #ffffff; color: #333333; border: 1px solid #e0e0e0; }
.stButton > button { background-color: #4caf50; color: white; border: none; border-radius: 8px; padding: 8px 16px; }
.stButton > button:hover { background-color: #388e3c; }
.st-info { background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 12px; border-radius: 8px; font-size: 14px; }
.stMarkdown { font-size: 16px; line-height: 1.6; }
.table-container { margin-top: 30px; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-header">🧠 Mental Wellness Chatbot</div>', unsafe_allow_html=True)
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

st.sidebar.title("Quick Start Prompts")
predefined_prompts = [
    "I'm feeling anxious about work. What can I do?",
    "How can I practice mindfulness daily?",
    "Begin the anxiety test and ask me the questions.",
    "What are some ways to build self-esteem?",
    "I feel overwhelmed. Help me prioritize."
]
for p in predefined_prompts:
    if st.sidebar.button(p):
        st.session_state.user_input = p

emergency_helplines = pd.DataFrame({
    "Country": ["USA", "India", "UK", "Canada", "Australia", "South Africa", "Japan"],
    "Emergency Helpline": ["911", "112", "999", "911", "000", "10111", "110"]
})
st.markdown("### Emergency Helplines Worldwide")
st.markdown("Below are some emergency helplines for countries around the world. In case of an emergency, please don't hesitate to reach out to the appropriate number.")
st.dataframe(emergency_helplines, use_container_width=True)

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables.")
    st.stop()

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

system_message = SystemMessage(content="""
You are a compassionate and supportive mental wellness assistant.
Offer practical coping strategies, mindfulness exercises, or general advice.
Remind users you are not a licensed therapist. Keep responses positive and empowering.
""".strip())

prompt_template = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

pipeline = prompt_template | llm

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history_msgs" not in st.session_state:
    st.session_state.history_msgs = [system_message]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if "user_input" in st.session_state:
    user_input = st.session_state.user_input
    del st.session_state.user_input
else:
    user_input = st.chat_input("How are you feeling today?")

MAX_HISTORY = 5

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.history_msgs.append(HumanMessage(content=user_input))
    if st.session_state.history_msgs and isinstance(st.session_state.history_msgs[0], SystemMessage):
        sys_msg = st.session_state.history_msgs[0]
        rest = st.session_state.history_msgs[1:]
        rest = rest[-MAX_HISTORY:]
        history_to_send = [sys_msg] + rest
    else:
        history_to_send = st.session_state.history_msgs[-MAX_HISTORY:]
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = pipeline.invoke({"input": user_input, "history": history_to_send})
            if not isinstance(response, str):
                response = str(response)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.history_msgs.append(AIMessage(content=response))
    if st.session_state.history_msgs and isinstance(st.session_state.history_msgs[0], SystemMessage):
        sys_msg = st.session_state.history_msgs[0]
        rest = st.session_state.history_msgs[1:]
        rest = rest[-MAX_HISTORY:]
        st.session_state.history_msgs = [sys_msg] + rest
    else:
        st.session_state.history_msgs = st.session_state.history_msgs[-MAX_HISTORY:]

