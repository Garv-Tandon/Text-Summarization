import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Simple Chatbot", page_icon="🤖")
st.title("🤖 Simple Chatbot")

# ---------------- Hugging Face API ----------------
hf_api_key = st.sidebar.text_input("🔑 Hugging Face API Token", type="password")

# ---------------- LLM ----------------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    token=hf_api_key,
    max_new_tokens=500,
    temperature=0.7
)

# ---------------- User Input ----------------
user_input = st.text_input("Type your question here:")

if st.button("Send"):
    if not hf_api_key:
        st.error("Please enter your Hugging Face API token.")
    elif not user_input:
        st.error("Please type a question.")
    else:
        # ✅ Proper call to HuggingFaceEndpoint
        llm_result = llm.generate([user_input])   # pass as list
        response = llm_result.generations[0][0].text

        st.markdown(f"**Bot:** {response}")
