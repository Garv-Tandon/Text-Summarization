import streamlit as st
from groq import Groq

st.set_page_config(page_title="Groq Chatbot", page_icon="🤖")
st.title("🤖 Simple Groq Chatbot")

# ---------------- Sidebar: API Key ----------------
api_key = st.sidebar.text_input("Groq API Key", type="password")

# ---------------- User Input ----------------
user_input = st.text_input("Ask a question:")

if st.button("Send"):
    if not api_key:
        st.error("Please enter your Groq API key.")
    elif not user_input:
        st.error("Please type a question.")
    else:
        try:
            client = Groq(api_key=api_key)

            # Send message to Groq model
            messages = [{"role": "user", "content": user_input}]
            response = client.chat.completions.create(
                messages=messages,
                model="openai/gpt-oss-20b"
            )

            # Display bot response
            st.markdown(f"**Bot:** {response.choices[0].message.content}")

        except Exception as e:
            st.error("❌ Error occurred")
            st.text(str(e))
