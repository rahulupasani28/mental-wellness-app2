import os
import streamlit as st
from langchain_groq import ChatGroq

# Load Groq API key from environment variable
API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    st.error("Groq API key not found. Please set the environment variable 'GROQ_API_KEY'.")
    st.stop()

# Instantiate LLM
llm = ChatGroq(
    groq_api_key=API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.4
)

# Streamlit page config
st.set_page_config(page_title="Light Theme Chatbot", page_icon="💬", layout="centered")
st.markdown("<h2 style='text-align: center; color: #2C666E;'>💬 Light Theme Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #333;'>Professional • Compassionate • Confidential</p>", unsafe_allow_html=True)
st.markdown("---")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Input message
message = st.text_area("Type your message here...", height=80)

# Send button
if st.button("Send"):
    if message.strip() != "":
        try:
            # Call Groq LLM
            response = llm.invoke(f"User: {message}\nRespond empathetically and professionally.")
            bot_reply = response.content.strip()
        except Exception:
            bot_reply = "⚠️ Sorry, I encountered an issue. Please try again later."
        
        # Save to chat history
        st.session_state.history.append({"user": message, "bot": bot_reply})

# Display chat
for chat in st.session_state.history:
    st.markdown(f"<div style='background-color: #2C666E; color: white; padding: 10px; border-radius: 8px; margin-bottom: 5px;'>You: {chat['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color: #DCEAE4; color: black; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>Bot: {chat['bot']}</div>", unsafe_allow_html=True)

# Footer note
st.markdown("---")
st.markdown("<small><b>Note:</b> This AI is supportive but not a replacement for professional help.</small>", unsafe_allow_html=True)
