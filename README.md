# 🤖 AI RAG Powered Profile Search 

This is a proof-of-concept (POC) application demonstrating a Retrieval-Augmented Generation (RAG) system for searching a database of professional profiles.

Instead of just using keywords, you can ask natural language questions (e.g., *"Find me a Python developer with 5 years of experience"*) and get a summarized, AI-generated answer.

This project was built using Streamlit, LangChain, Groq, and ChromaDB to showcase the core AI logic before integration into a larger production system.

## 🚀 Live Demo

**You can test the live application here:**

https://ai-profile-search-by-rag-model-manikanta.streamlit.app/

## ✨ Features

* **Natural Language Queries:** Ask questions in plain English.
* **AI-Powered Summaries:** Get a clean, summarized answer from the high-speed Groq Llama 3.1 model.
* **Grounded & Verifiable:** All answers are based *only* on the provided data (`profiles.csv`). You can expand the "Show relevant profile chunks" section to verify the sources and prevent AI "hallucinations."

## 🛠️ Tech Stack

* **UI:** Streamlit
* **LLM:** Groq (using `llama-3.1-8b-instant`)
* **Framework:** LangChain (using the `RetrievalQA` chain)
* **Vector Database:** ChromaDB (runs locally, persists in the `vector_db` folder)
* **Embeddings:** Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)

## ⚙️ How it Works (RAG Architecture)

1.  **Ingestion (First Run):** The first time the app starts, it checks for a `vector_db` folder. If it's missing, the app automatically reads `profiles.csv`, splits the data into chunks, and uses the Hugging Face model to create vector embeddings. These are stored in the local `vector_db` folder.
2.  **Retrieval:** When you ask a question, the app converts your query into a vector and searches the `vector_db` to find the most relevant profile chunks.
3.  **Generation:** The app sends your question *and* the retrieved chunks to the Groq LLM with a prompt, instructing it to answer the question *only* using the provided context.

## 🏃 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    # Create the venv
    python -m venv venv
    
    # Activate the venv
    venv\Scripts\activate  # On Windows (PowerShell/CMD)
    # source venv/bin/activate  # On macOS/Linux
    
    # Install libraries
    pip install -r requirements.txt
    ```

3.  **Set your Groq API Key:**
    
    *On Windows (PowerShell):*
    ```bash
    $Env:GROQ_API_KEY = "gsk_...your...key...here..."
    ```
    
    *On macOS/Linux:*
    ```bash
    export GROQ_API_KEY="gsk_...your...key...here..."
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The app will open in your browser. The *first time* you run it, it will automatically build the `vector_db` folder (this may take a minute).

## ☁️ Deployment

This app is deployed on [Streamlit Community Cloud](https://share.streamlit.io/). To deploy your own version:

1.  Push the repository (containing `app.py`, `requirements.txt`, and `profiles.csv`) to your own public GitHub account.
2.  Connect your GitHub to Streamlit Community Cloud.
3.  In the app's **"Advanced settings..."**, add your Groq API key to the **Secrets** section:
    ```toml
    GROQ_API_KEY = "gsk_...your...key...here..."
    ```
4.  Click **"Deploy!"**

## 📁 File Structure
  . ├── 📄 app.py # The main Streamlit application
  
  . ├── 📄 profiles.csv # The sample dataset of profiles 
  
  . ├── 📄 requirements.txt # Python libraries needed to run the project 
  
  . └── 📁 vector_db/ # (Generated automatically) The persistent vector database
