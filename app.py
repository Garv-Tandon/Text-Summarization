import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Simple Chatbot", page_icon="🤖")
st.title("🤖 Simple Chatbot with LangChain")

# ---------------- Sidebar: Hugging Face API ----------------
hf_api_key = st.sidebar.text_input("🔑 Hugging Face API Token", type="password")

# ---------------- HuggingFace LLM ----------------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    token=hf_api_key,
    max_length=500,
    temperature=0.7
)

# ---------------- Prompt Template ----------------
chat_prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Answer the following question:\n{question}"
)

# ---------------- Chat Input ----------------
user_input = st.text_input("Ask me anything:")

if st.button("Send"):
    if not hf_api_key:
        st.error("Please enter your Hugging Face API token.")
    elif not user_input:
        st.error("Please type a question.")
    else:
        prompt_text = chat_prompt.format(question=user_input)
        response = llm(prompt_text)
        st.markdown(f"**Bot:** {response}")
