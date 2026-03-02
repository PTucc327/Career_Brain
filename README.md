# 🧠 Paul's AI Career Brain (Local RAG Pipeline)

An interactive, local AI agent capable of answering technical questions about my professional experience and codebase. This project uses **Retrieval-Augmented Generation (RAG)** to provide grounded, accurate responses based on my actual work history.

## 🛠️ Technical Stack
* **Orchestration:** [LlamaIndex](https://www.llamaindex.ai/)
* **Local LLM:** [Ollama](https://ollama.com/) (Llama 3.2 1B)
* **Embeddings:** `nomic-embed-text` (Local)
* **UI Framework:** [Streamlit](https://streamlit.io/)
* **Data Sources:** GitHub API (Repositories) & Local PDF (Resume)

## 🏗️ Architecture
The system follows a standard RAG pattern but is optimized for local execution on consumer hardware:
1.  **Ingestion:** Scrapes Python and Jupyter Notebook files from specific GitHub repos using `GithubRepositoryReader`.
2.  **Indexing:** Vectorizes code and text into a local document store.
3.  **Retrieval:** Uses a `VectorStoreIndex` to fetch the top 5 most relevant context chunks for any query.
4.  **Generation:** Passes context to a locally hosted Llama 3.2 model with a custom system prompt to act as a "Career Advocate."

## 🚀 Key Features
* **100% Local & Private:** No data leaves the machine; no OpenAI API keys required.
* **Code-Aware:** Can identify specific libraries (Scikit-Learn, Pandas, etc.) used in my projects.
* **Streaming UI:** Real-time response generation for a smooth user experience.