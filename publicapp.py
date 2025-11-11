import streamlit as st
from langchain_groq import ChatGroq
import os
import pandas as pd

# --- All Imports ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
DB_PATH = "vector_db"
DATA_PATH = "profiles.csv" # Your CSV file must be in GitHub too
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

# --- Function to Build the Vector DB ---
def build_vector_db():
    """Builds the vector database from the CSV file."""
    st.info("Vector database not found. Building a new one...")
    
    loader = CSVLoader(file_path=DATA_PATH, encoding="utf-8")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=DB_PATH
    )
    
    st.success(f"Vector database built successfully! {len(chunks)} chunks indexed.")
    return db

# --- Main App Logic ---

# --- Initialize Models ---
@st.cache_resource
def load_models():
    """Loads embedding model and vector DB. Builds DB if not found."""
    try:
        # Check if the vector DB exists.
        if not os.path.exists(DB_PATH):
            build_vector_db()

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        
        retriever = db.as_retriever(search_kwargs={'k': 3})
        llm = ChatGroq(model_name="llama-3.1-8b-instant")
        
        return retriever, llm

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# --- Load API Key from Streamlit Secrets ---
# For deployment, you can't use $Env.
# You must set this in Streamlit Cloud's "Secrets" menu.
if "GROQ_API_KEY" not in os.environ:
    try:
        # Check Streamlit's secrets
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except:
        st.error("GROQ_API_KEY is not set. Please add it to your Streamlit Cloud secrets.")
        st.stop()

# --- Load Models ---
retriever, llm = load_models()

# --- Build RAG QA Chain ---
if retriever and llm:
    # Prompt Template
    prompt_template = """
    You are an expert HR assistant. Your ONLY task is to answer user questions 
    based *strictly* and *exclusively* on the context provided.

    Use the following retrieved context to answer the user's question:
    
    Context:
    {context}
    
    Question:
    {question}

    Rules you MUST follow:
    1.  Use ONLY the context. Do not use any of your outside knowledge.
    2.  If the answer is not in the context, you MUST say "I do not have enough 
        information to answer this question."
    3.  Do NOT make suggestions, offer alternatives, or add any information 
        that is not explicitly in the context.
    4.  Provide concise summaries for profiles you find.

    Answer:
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": ChatPromptTemplate.from_template(prompt_template)
        },
        return_source_documents=True
    )

    # --- Streamlit UI ---
    st.title("🤖 AI Profile Search (RAG Demo)")
    st.write("Ask me to find profiles, e.g., 'Who is a data scientist?'")

    query = st.text_input("Enter your search query:", key="search_query")

    if st.button("Search", key="search_button"):
        if query:
            with st.spinner("Searching profiles..."):
                try:
                    response = qa_chain.invoke(query) 
                    st.subheader("AI-Generated Answer")
                    st.write(response["result"])
                    with st.expander("🔍 Show relevant profile chunks (Context)"):
                        for doc in response["source_documents"]:
                            st.markdown(f"**Source:** {doc.metadata}\n\n{doc.page_content}\n---")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query.")
else:

    st.info("Models are not loaded. Please check your setup.")
