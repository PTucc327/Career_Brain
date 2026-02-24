import streamlit as st
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.github import GithubRepositoryReader, GithubClient
import os
from dotenv import load_dotenv

# 1. Page Configuration
st.set_page_config(page_title="Paul's Career Brain", page_icon="🤖", layout="centered")
st.title("🤖 Paul's Career Brain")
st.markdown("Ask me anything about Paul's technical projects and experience.")

# 2. Setup Local AI & Data (Cached so it only runs once)
@st.cache_resource
def initialize_brain():
    load_dotenv()
    # Using the 1b model for speed and memory efficiency
    Settings.llm = Ollama(
    model="llama3.2:1b", 
    request_timeout=120.0,
    additional_kwargs={"keep_alive": 0} 
    )

    Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    additional_kwargs={"keep_alive": 0} 
    )
    
    github_client = GithubClient(os.getenv("GITHUB_TOKEN"))

    repos = [
    {"owner": "PTucc327", "repo": "ChurnLens"},
    {"owner": "PTucc327", "repo": "Job_Market_Analysis"},
    {"owner": "PTucc327", "repo": "NFL_Pass_Rush_Play_Type_Classification"}
    ]
    
    all_docs = []
    for repo in repos:
        loader = GithubRepositoryReader(
            github_client, owner=repo["owner"], repo=repo["repo"],
            filter_file_extensions=([".py", ".ipynb", ".md", ".r", ".Rmd"], GithubRepositoryReader.FilterType.INCLUDE)
        )
        all_docs.extend(loader.load_data(branch="main"))
    
    return VectorStoreIndex.from_documents(all_docs).as_query_engine()

# Initialize the engine
with st.spinner("🧠 Waking up the brain... this may take a moment."):
    query_engine = initialize_brain()

# 3. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What would you like to know about Paul?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = query_engine.query(prompt)
        st.markdown(str(response))
        st.session_state.messages.append({"role": "assistant", "content": str(response)})