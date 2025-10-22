import streamlit as st
from transformers import pipeline

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Offline Chatbot", page_icon="🤖")
st.title("🤖 Offline Chatbot")
st.write("This chatbot runs locally—no API key needed!")

# ---------------- Load Local Model ----------------
@st.cache_resource(show_spinner=True)
def load_model():
    return pipeline("text-generation", model="distilgpt2")

generator = load_model()

# ---------------- Chat Input ----------------
user_input = st.text_input("Type your message:")

if st.button("Send"):
    if not user_input:
        st.error("Please type a message.")
    else:
        # Generate response
        output = generator(user_input, max_length=100, do_sample=True, top_k=50)
        st.markdown(f"**Bot:** {output[0]['generated_text']}")
