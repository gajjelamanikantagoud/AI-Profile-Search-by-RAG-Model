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

# --- 1. PAGE CONFIGURATION (Makes it look professional) ---
st.set_page_config(
    page_title="AI Profile Search",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM STYLING (Hides Streamlit "extras") ---
st.markdown("""
<style>
    /* Hide the 'Made with Streamlit' footer */
    footer {visibility: hidden;}
    /* Hide the Streamlit main menu */
    #MainMenu {visibility: hidden;}
    /* Clean up the top margin */
    .block-container {
        padding-top: 2rem;
    }
    /* Style for the answer box */
    .answer-container {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    /* Style for source document cards */
    .source-card {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration (Your code) ---
DB_PATH = "vector_db"
DATA_PATH = "profiles.csv" # Your CSV file must be in GitHub too
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

# --- Function to Build the Vector DB (Your code) ---
def build_vector_db():
    """Builds the vector database from the CSV file."""
    with st.spinner("Vector database not found. Building a new one... This may take a minute."):
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

# --- Initialize Models (Your code) ---
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

# --- Load API Key from Streamlit Secrets (Your code) ---
if "GROQ_API_KEY" not in os.environ:
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except:
        st.error("GROQ_API_KEY is not set. Please add it to your Streamlit Cloud secrets or set it as an environment variable locally.")
        st.stop()

# --- Load Models ---
retriever, llm = load_models()

# --- Build RAG QA Chain ---
if retriever and llm:
    # --- 3. IMPROVED (Stricter) PROMPT ---
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
    2.  Do NOT make suggestions, offer alternatives, or add any information 
        that is not explicitly in the context.
    3.  Provide concise summaries for profiles you find.

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

    # --- 4. NEW & IMPROVED STREAMLIT UI ---
    
    # Title and Subtitle
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🤖 AI Profile Search by RAG</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555;'>Ask a natural language question about our applicant profiles below.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Centered Search Bar
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        query = st.text_input("Enter your search query:", key="search_query", placeholder="e.g., 'Find me a data scientist with Tableau skills'")
        search_button = st.button("Search Profiles", use_container_width=True, type="primary")

    st.markdown("---")

    # Results Area
    if search_button and query:
        with st.spinner("Searching profiles and generating answer..."):
            try:
                response = qa_chain.invoke(query) 
                
                # Create two columns for Answer and Sources
                col_ans, col_src = st.columns(2)
                
                with col_ans:
                    st.subheader("💡 AI-Generated Answer")
                    # Use st.container with a border for a "card" effect
                    with st.container(border=True):
                        st.write(response["result"])

                with col_src:
                    st.subheader("🔍 Source Documents")
                    # Display sources in their own "card" containers
                    for doc in response["source_documents"]:
                        with st.container(border=True):
                            source_info = doc.metadata.get('source', 'N/A')
                            row_info = doc.metadata.get('row', 'N/A')
                            st.markdown(f"**Source:** `{source_info}` (Row: `{row_info}`)")
                            
                            # Show a snippet instead of the full text
                            st.markdown(f"> {doc.page_content[:200]}...")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
    elif not query and search_button:
        st.warning("Please enter a query.")
else:
    st.info("Models are not loaded. Please check your setup.")
