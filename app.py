import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.github import GithubRepositoryReader, GithubClient

# 1. Page Configuration
st.set_page_config(page_title="Paul's Career Brain", page_icon="🤖", layout="centered")
st.title("🤖 Paul's Career Brain")
st.markdown("Ask me anything about Paul's technical projects and experience.")

# 2. Setup Local AI & Data (Cached)
@st.cache_resource
def initialize_brain():
    load_dotenv()
    
    # Memory-optimized settings
    Settings.llm = Ollama(model="llama3.2:1b", request_timeout=120.0, additional_kwargs={"keep_alive": 0})
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", additional_kwargs={"keep_alive": 0})
    
    # Load Resume
    resume_reader = SimpleDirectoryReader("./data")
    resume_docs = resume_reader.load_data()

    # Initialize GitHub Client
    github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
    repos = [
        {"owner": "PTucc327", "repo": "ChurnLens"},
        {"owner": "PTucc327", "repo": "Job_Market_Analysis"},
        {"owner": "PTucc327", "repo": "NFL_Pass_Rush_Play_Type_Classification"}
    ]
    
    repo_docs = []
    for repo in repos:
        loader = GithubRepositoryReader(
            github_client, owner=repo["owner"], repo=repo["repo"],
            filter_file_extensions=([".py", ".ipynb", ".md"], GithubRepositoryReader.FilterType.INCLUDE),
            verbose=True 
        )
        repo_docs.extend(loader.load_data(branch="main"))
    
    all_docs = resume_docs + repo_docs
    index = VectorStoreIndex.from_documents(all_docs)
    
    # Setup the prompt inside the engine
    system_prompt = (
        "You are the AI version of Paul Tuccinardi's Career Brain. "
        "Your goal is to represent Paul to recruiters. Use the provided context to answer questions specifically. "
        "Mention specific libraries like sklearn, selenium, or pandas if found in his code. "
        "Be professional, confident, and helpful."
    )
    
    return index.as_query_engine(system_prompt=system_prompt, similarity_top_k=5,streaming=True)

# --- THIS IS THE KEY CHANGE ---
with st.spinner("🧠 Waking up the brain... this may take a moment."):
    # We call the function and save the result into 'query_engine'
    query_engine = initialize_brain() 

# 3. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know about Paul?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
    # This now returns a StreamingResponse object
        response_stream = query_engine.query(prompt)
        
        # st.write_stream iterates through the response_gen for you
        full_response = st.write_stream(response_stream.response_gen)
        
        # Save the full string to your session history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        
with st.sidebar:
    st.image("https://github.com/PTucc327.png") # Pulls your GitHub profile pic automatically!
    st.title("Paul Tuccinardi")
    st.markdown("📍 **Location:** Stamford, CT")
    st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/paul-tuccinardi/)")
    st.markdown("💻 [GitHub](https://github.com/PTucc327)")
    
    # Add a download button for your actual PDF
    with open("./data/Paul_Tuccinardi.pdf", "rb") as f:
        st.download_button(
            label="📄 Download My Full Resume",
            data=f,
            file_name="Paul_Tuccinardi_Resume.pdf",
            mime="application/pdf"
        )