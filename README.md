# 🧠 Paul's AI Career Brain (Local RAG Pipeline)

An interactive, local AI agent capable of answering technical questions about my professional experience and codebase. This project uses **Retrieval-Augmented Generation (RAG)** to provide grounded, accurate responses based on my actual work history.

## 🛠️ Technical Stack
* **Orchestration:** [LlamaIndex](https://www.llamaindex.ai/)
* **Local LLM:** [Ollama](https://ollama.com/) (Llama 3.2 1B)
* **Embeddings:** `OllamaEmbedding` (Llama 3.2)
* **UI Framework:** [Streamlit](https://streamlit.io/)
* **Data Sources:** GitHub API (Repositories) & Local PDF (Resume)

## 🏗️ Architecture
The system follows a standard RAG pattern optimized for local execution:
1.  **Ingestion:** Scrapes Python and Jupyter Notebook files from GitHub using `GithubRepositoryReader` and parses local PDFs.
2.  **Indexing:** Vectorizes code and text into a high-dimensional document store.
3.  **Retrieval:** Uses a `VectorStoreIndex` to fetch the most relevant context chunks for any query.
4.  **Generation:** Passes context to a locally hosted Llama 3.2 model with a custom system prompt to act as a "Career Advocate."



## 🚀 Key Features
* **100% Local & Private:** No data leaves the machine; powered by Ollama.
* **Code-Aware:** Specifically indexed to identify my work with Scikit-Learn, PyTorch, and XGBoost.
* **Streaming UI:** Real-time token generation for a smooth, low-latency user experience.
* **Persistent Memory:** Uses `ChatMemoryBuffer` to maintain context during multi-turn interviews.

## 🎯 What the Brain Knows
You can ask the AI about specific milestones in my portfolio, such as:
* **NFL RAG Chatbot:** Ask about the LlamaIndex/Pinecone architecture.
* **Succession AI R&D:** Details on automating financial summaries using 7+ LLMs.
* **Job Market Analysis:** My work with SBERT and BERT-based embeddings.

## ⚙️ Setup & Installation
1. **Clone the repo:**
   ```bash
   git clone https://github.com/PTucc327/Career_Brain.git
   cd Career_Brain
   ```
2. **Install Dependencies**
    ```bash
    pip install requirements.txt
    ```
3. **Configure Environment**
    - Rename template.env to .env and add your GITHUB_TOKEN.
4. **Ensure Ollama is running**
    ```bash
    ollama pull llama3.2:1b
    ```
5. **Run Streamlit app**
    ```bash
    streamlit run app.py
    ```
    