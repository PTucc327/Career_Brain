import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.memory import ChatMemoryBuffer

# --- 1. Page Configuration & Custom CSS ---
st.set_page_config(page_title="Paul's Career Brain", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #0078D4; color: white; }
    .stDownloadButton>button { width: 100%; border-radius: 5px; background-color: #28a745; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Initialize Models & Settings ---
@st.cache_resource
def configure_settings():
    # request_timeout increased to 300s to prevent httpx.ReadTimeout during long RAG retrievals
    Settings.llm = Ollama(model="llama3.2:1b", request_timeout=300.0, temperature=0.1)
    Settings.embed_model = OllamaEmbedding(model_name="llama3.2:1b")

configure_settings()

# --- 3. Data Ingestion & Brain Initialization ---
@st.cache_resource
def initialize_brain():
    # Load documents from your /data folder (Resume, LinkedIn text, etc.)
    if not os.path.exists("./data"):
        st.error("Data folder not found. Please ensure your PDF and text files are in the /data directory.")
        return None

    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    # Initialize Memory
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    
    # Create Chat Engine with the "Persona" instruction
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        similarity_top_k=3,
        system_prompt=(
    "You are Paul Tuccinardi's AI representative. You are grounded, technical, and precise."
    "\n\nSTRICT GUIDELINES:"
    "\n1. ONLY use the provided context. If an answer isn't there, say: 'I don't have that specific data in my records.'"
    "\n2. NEVER mention page numbers (e.g., 'On page 2...'). Instead, say 'According to Paul's resume...' or 'In the NFL project documentation...'"
    "\n3. Do not assume skills. If asked about Kubernetes, say it is not listed, but highlight Docker experience instead."
    "\n4. Keep projects separate: "
    "\n   - Job Market Analysis: SBERT, XGBoost, Python."
    "\n   - NFL Classification: XGBoost, FastAPI, 97% F1-score."
    "\n   - Charter: Tableau, SQL, Geospatial analysis."
    "\n5. If the user asks a follow-up, use the chat history to maintain context."
    "\n6. TECHNICAL ACCURACY: SBERT is an open-source framework (Hugging Face/UKP Lab), not an OpenAI product. Do not attribute open-source tools to OpenAI unless explicitly stated."
)
    )
    return chat_engine

chat_engine = initialize_brain()

# --- 4. Sidebar UI ---
with st.sidebar:
    # Adding a clean header
    st.image("https://github.com/PTucc327.png")
    st.title("Paul Tuccinardi")
    st.text("🚀 Data Scientist | Data Analyst | ML Engineer")
    st.text("📍 Stamford, CT ")
    st.text("📧 paultuccinardi@gmail.com")
    
    
    c1, c2 = st.columns(2)
    with c1: st.link_button("LinkedIn", "https://linkedin.com/in/paultuccinardi/", use_container_width=True)
    with c2: st.link_button("GitHub", "https://github.com/PTucc327", use_container_width=True)

    st.info("Ask my AI twin about my professional background, technical skills, or specific projects.")

    st.divider()
    
    # Ensure this file path matches your uploaded resume name exactly
    resume_path = "./data/Paul_Tuccinardi.pdf"
    if os.path.exists(resume_path):
        with open(resume_path, "rb") as f:
            st.download_button("📄 Download Resume", f, file_name="Paul_Tuccinardi_Resume.pdf", mime="application/pdf")

    st.divider()
    st.subheader("🎯 Featured Topics")
    st.markdown("""
    - **NFL Chatbot** (LlamaIndex/Pinecone)
    - **Fitbit Dashboard** (Sensor Data)
    - **Job Market Analysis** (SBERT/XGBoost)
    - **Succession AI** (LLM Automation)
    """)

# --- 5. Main Chat Interface (with Streaming) ---
st.title("🤖 Career Brain")
st.toast("Welcome! How can I help you learn more about Paul today?", icon="👋")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm Paul's digital twin. Ask me anything about my work at Succession AI, my NFL analytics projects, or my time at Pace University."}
    ]

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Logic
if prompt := st.chat_input("Ask me about Paul's experience..."):
    # 1. Add user message to state and UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate and stream assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Use stream_chat instead of chat for real-time output
        streaming_response = chat_engine.stream_chat(prompt)
        
        # Loop through tokens as they arrive from Ollama
        for token in streaming_response.response_gen:
            full_response += token
            # Show a typing cursor while streaming
            response_placeholder.markdown(full_response + "▌")
        
        # Final display without the cursor
        response_placeholder.markdown(full_response)
        
        # Save final response to state
        st.session_state.messages.append({"role": "assistant", "content": full_response})