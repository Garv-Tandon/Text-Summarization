import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Simple Chatbot", page_icon="🤖")
st.title("🤖 Simple Hugging Face Chatbot")

# Hugging Face API Key
hf_api_key = st.sidebar.text_input("🔑 Hugging Face API Token", type="password")

# User input
user_input = st.text_input("Type your question here:")

if st.button("Send"):
    if not hf_api_key:
        st.error("Please enter your Hugging Face API token.")
    elif not user_input:
        st.error("Please type a question.")
    else:
        try:
            client = InferenceClient(token=hf_api_key)
            # Use Mistral 7B Instruct model
            response = client.text_generation(
                model="mistralai/Mistral-7B-Instruct-v0.3",
                inputs=user_input,
                parameters={"max_new_tokens": 500, "temperature": 0.7}
            )
            st.markdown(f"**Bot:** {response[0]['generated_text']}")
        except Exception as e:
            st.error("❌ Error occurred")
            st.text(str(e))
