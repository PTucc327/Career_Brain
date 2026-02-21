import os
from dotenv import load_dotenv
# Corrected Imports for v0.10+
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# 1. Load your variables
load_dotenv()
github_token = os.getenv("GITHUB_TOKEN")

# 2. Setup LOCAL AI (Ollama) - No OpenAI needed!
# This tells LlamaIndex to use the models you 'pulled' in the terminal
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# 3. Initialize GitHub Client
github_client = GithubClient(github_token)

# 4. Define your Projects
repos = [
    {"owner": "PTucc327", "repo": "ChurnLens"},
    {"owner": "PTucc327", "repo": "Job_Market_Analysis"},
    {"owner": "PTucc327", "repo": "NFL_Pass_Rush_Play_Prediction"}
]

all_docs = []

# 5. Load data from GitHub
print("📥 Fetching code from GitHub...")
for repo in repos:
    loader = GithubRepositoryReader(
        github_client,
        owner=repo["owner"],
        repo=repo["repo"],
        filter_file_extensions=([".py", ".ipynb", ".md"], GithubRepositoryReader.FilterType.INCLUDE)
    )
    all_docs.extend(loader.load_data(branch="main"))

# 6. Build the Local Index
print("🧠 Processing data into your local brain (this may take a minute)...")
index = VectorStoreIndex.from_documents(all_docs)

# 7. Query it!
query_engine = index.as_query_engine()
response = query_engine.query("Based on the ChurnLens and NFL_Pass_Rush_Play_Prediction projects, what is Paul's strongest programming language?")
print(f"\nResult: {response}")