import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain.docstore.document import Document
from pytube import YouTube
from langchain_community.document_loaders import UnstructuredURLLoader
import traceback

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# ---------------- Sidebar API Key ----------------
with st.sidebar:
    hf_api_key = st.text_input("Huggingface API Token", value="", type="password")

# ---------------- URL Input ----------------
generic_url = st.text_input("URL", label_visibility="collapsed")

# ---------------- HuggingFace Model ----------------
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=500,
    temperature=0.7,
    token=hf_api_key
)

# ---------------- Summarization Prompts ----------------
map_prompt_template = "Summarize the following text:\n{text}"
combine_prompt_template = "Given the following summaries, combine them into a comprehensive summary of 300 words:\n{text}"

map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

# ---------------- Helper Functions ----------------
def extract_youtube_id(url):
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(url)
    if parsed.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed.query).get("v", [None])[0]
    elif parsed.hostname in ["youtu.be"]:
        return parsed.path[1:]
    return None

def fetch_youtube_captions(video_url):
    try:
        yt = YouTube(video_url)
        caption = yt.captions.get_by_language_code("en")  # English captions
        if caption:
            text = caption.generate_srt_captions()
            return text
        else:
            return None
    except Exception as e:
        st.error("Failed to fetch YouTube captions.")
        st.text(str(e))
        st.text(traceback.format_exc())
        return None

# ---------------- Summarization Button ----------------
if st.button("Summarize the Content from YT or Website"):
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the API key and a URL to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or website URL.")
    else:
        try:
            docs = []
            with st.spinner("Loading content..."):

                # --- YouTube ---
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    text = fetch_youtube_captions(generic_url)
                    if not text:
                        st.error("No captions available for this video.")
                        st.stop()
                    docs.append(Document(page_content=text))

                # --- Website ---
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                if not docs:
                    st.error("No content could be extracted from the URL.")
                    st.stop()

                # --- Summarization Chain ---
                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt
                )
                output_summary = chain.run(docs)

                st.success("✅ Summary Generated!")
                st.write(output_summary)

                # --- Download Summary Button ---
                st.download_button(
                    label="Download Summary",
                    data=output_summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )

        except Exception as e:
            st.error("An error occurred during summarization.")
            st.text(str(e))
            st.text(traceback.format_exc())
