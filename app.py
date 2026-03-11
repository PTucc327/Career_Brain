import streamlit as st
import os
import requests
import urllib.parse
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.core.memory import ChatMemoryBuffer
from streamlit_mic_recorder import speech_to_text

# 1. Page Config
st.set_page_config(page_title="Paul's Career Brain", page_icon="🤖", layout="wide")

@st.cache_resource
def initialize_brain():
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    github_username = "PTucc327"
    paul_email = "paultuccinardi@gmail.com"
    linkedin_url = "https://www.linkedin.com/in/paultuccinardi/" # Your profile
    
    # --- GitHub Data Fetching ---
    headers = {"Authorization": f"token {github_token}"}
    all_repos_raw = requests.get(f"https://api.github.com/users/{github_username}/repos?per_page=100", headers=headers).json()
    
    query = """{ user(login: "%s") { pinnedItems(first: 6, types: REPOSITORY) { nodes { ... on Repository { name url } } } } } """ % github_username
    try:
        gql_res = requests.post("https://api.github.com/graphql", json={'query': query}, headers={"Authorization": f"Bearer {github_token}"}).json()
        pinned_names = [node['name'] for node in gql_res['data']['user']['pinnedItems']['nodes']]
    except: pinned_names = []

    # --- AI Setup (Ollama) ---
    Settings.llm = Ollama(model="llama3.2:1b", temperature=0.1, request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    # --- DATA LOADING ---
    # A. Resume
    resume_docs = SimpleDirectoryReader("./data").load_data()
    
    # B. LinkedIn (Public Profile Reader)
    # Note: This works best if your profile visibility is set to 'Public'
    web_reader = BeautifulSoupWebReader()
    linkedin_docs = web_reader.load_data(urls=[linkedin_url])
    
    # C. GitHub Code
    github_client = GithubClient(github_token)
    repo_docs, processed_metadata = [], []

    for r in all_repos_raw:
        if r["fork"] or r["name"].startswith("."): continue
        processed_metadata.append({"name": r["name"], "url": r["html_url"], "desc": r["description"], "is_pinned": r["name"] in pinned_names})
        try:
            loader = GithubRepositoryReader(github_client, owner=github_username, repo=r["name"], filter_file_extensions=([".py", ".ipynb", ".md"], GithubRepositoryReader.FilterType.INCLUDE))
            repo_docs.extend(loader.load_data(branch=r["default_branch"]))
        except: continue

    # Combine all knowledge sources
    all_docs = resume_docs + linkedin_docs + repo_docs
    index = VectorStoreIndex.from_documents(all_docs)
    
    # --- Chat Engine with Persona & Memory ---
    memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=(
            f"You are Paul Tuccinardi's AI double. Paul's email is {paul_email}.\n"
            "Identity: Paul is a recent graduate from Pace University with an M.S. in Data Science and B.S. in Computer Science.\n"
            "Context Sources: GitHub (Code), PDF Resume (History), and LinkedIn (Professional Summary).\n\n"
            "STRICT RULES:\n"
            "1. If a user asks to contact Paul, draft a professional email starting with 'DRAFT:'.\n"
            "2. Distinguish projects: GitHub projects are either Coursework or Independent Data Science projects.\n"
            "3. Mention Pace University specifically regarding his degrees and relevant academic rigor.\n"
            "4. For technical questions, link to the relevant GitHub URL from the context.\n"
            "5. Keep responses concise, helpful, and professional."
        )
    )
    return chat_engine, processed_metadata, paul_email

with st.spinner("🧠 Syncing Paul's Neural Network (GitHub + Resume + LinkedIn)..."):
    chat_engine, repo_metadata, PAUL_EMAIL = initialize_brain()

# --- SIDEBAR UI ---
with st.sidebar:
    st.image("https://github.com/PTucc327.png", width=120)
    st.title("Paul Tuccinardi")
    st.write("📍 Stamford, CT | 🚀 Data Scientist & Data Analyst")
    
    c1, c2 = st.columns(2)
    with c1: st.link_button("LinkedIn", "https://linkedin.com/in/paultuccinardi/", use_container_width=True)
    with c2: st.link_button("GitHub", "https://github.com/PTucc327", use_container_width=True)

    st.markdown("---")
    st.subheader("📌 Featured Projects")
    for repo in [r for r in repo_metadata if r["is_pinned"]]:
        with st.expander(f"**{repo['name'].replace('_', ' ')}**"):
            st.caption(repo['desc'] or "Technical deep-dive.")
            st.page_link(repo['url'], label="Source Code", icon="🔗")

# --- MAIN CHAT ---
st.title("🤖 Paul's Career Brain")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Multi-modal Input
col_v, col_t = st.columns([0.1, 0.9])
with col_v: 
    v_input = speech_to_text(language='en', start_prompt="🎤", stop_prompt="⏹️", just_once=True, key='STT')
with col_t: 
    t_input = st.chat_input("Ask about my NFL projects, LinkedIn summary, or draft an email...")

prompt = v_input if v_input else t_input

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chat_engine.chat(prompt).response
        st.markdown(response)
        
        # Email Drafting Logic
        if "DRAFT:" in response.upper():
            body_text = response.replace("DRAFT:", "").strip()
            subject = "Reaching out regarding Paul's Portfolio"
            mailto_url = f"mailto:{PAUL_EMAIL}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body_text)}"
            
            st.markdown(f"""
                <a href="{mailto_url}" target="_blank">
                    <button style="background-color: #ff4b4b; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
                        📧 Open in Email Client
                    </button>
                </a>
            """, unsafe_allow_html=True)

        st.session_state.chat_history.append({"role": "assistant", "content": response})