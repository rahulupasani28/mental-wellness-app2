import os
import gradio as gr
from langchain_groq import ChatGroq

# Load API key from environment variable
API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("Groq API key not found. Please set the environment variable 'GROQ_API_KEY'.")

# Chatbot class
class LightThemeChatbot:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.4
        )

    def chat(self, message, history):
        try:
            response = self.llm.invoke(f"User: {message}\nRespond empathetically and professionally.")
            return response.content.strip()
        except Exception as e:
            return "⚠️ Sorry, I encountered an issue. Please try again later."

# Instantiate chatbot
bot = LightThemeChatbot()

# Create Gradio app
def create_app():
    def chat_fn(message, history):
        return bot.chat(message, history)

    # Light theme CSS
    light_css = """
    body, .gradio-container {
        background: #ffffff !important;
        color: #222222 !important;
    }
    .main-card {
        background: #ffffff !important;
        border: 1px solid #ddd !important;
        border-radius: 10px !important;
        padding: 2rem !important;
    }
    .chatbot { background: #f9f9f9 !important; }
    .message.user { background: #2C666E !important; color: #ffffff !important; }
    .message.bot { background: #DCEAE4 !important; color: #000000 !important; }
    """

    with gr.Blocks(theme=gr.themes.Soft(), css=light_css, title="Light Theme Chatbot") as app:
        with gr.Column(elem_classes="main-card"):
            gr.Markdown("## 💬 Light Theme Chatbot")
            gr.Markdown("Professional • Compassionate • Confidential")

            gr.ChatInterface(
                fn=chat_fn,
                title="🌟 Share Your Thoughts",
                description="A simple AI companion in light theme.",
                examples=[
                    "I feel anxious about exams.",
                    "I had a rough day today.",
                    "I'm feeling better but still a little stressed."
                ],
                chatbot=gr.Chatbot(height=450, bubble_full_width=False, render_markdown=True),
                textbox=gr.Textbox(placeholder="Type your message here...", lines=2),
                type="messages"
            )

            gr.Markdown("---")
            gr.Markdown(
                "<small><b>Note:</b> This AI is supportive but not a replacement for professional help.</small>",
                elem_classes="footer"
            )
    return app

# Launch app
app = create_app()
app.launch(share=True)
