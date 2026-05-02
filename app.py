import streamlit as st
import os
import tempfile
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. PAGE CONFIGURATION
# Set up the web interface title and wide-screen layout for better readability of comparisons.
st.set_page_config(page_title="Cross-Ref Engine", layout="wide")

# 2. SIDEBAR SETUP (Input Controls)
with st.sidebar:
    st.title("⚙️ Setup")
    # API Key is masked for security. 
    api_key = st.text_input("Enter Google API Key", type="password")
    # Allow multiple uploads to enable the "Cross-Reference" capability between different books.
    uploaded_files = st.file_uploader("Upload Textbooks (PDF)", type="pdf", accept_multiple_files=True)

st.title("📚 Academic Cross-Reference Engine")
st.write("Synthesize information and find contradictions between multiple academic sources.")

# 3. CORE LOGIC (RAG Pipeline)
if uploaded_files and api_key:
    # Set environment variable for Google Generative AI authentication
    os.environ["GOOGLE_API_KEY"] = api_key
    status = st.empty() # Placeholder for real-time status updates
    
    # --- STEP A: DATA INGESTION ---
    # Load all PDF pages into a centralized list. We use a temporary file to handle 
    # the uploaded buffer data safely before passing it to PyPDFLoader.
    all_pages = []
    status.info("📖 Reading documents...")
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            all_pages.extend(loader.load())

    # --- STEP B: TEXT CHUNKING ---
    # Break long textbooks into smaller pieces (1000 chars) so the AI can find specific facts.
    # The 100-character overlap ensures that context is not lost between chunks.
    status.info("✂️ Splitting text into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_pages)

    # --- STEP C: RESILIENT VECTOR SYNCING (RAG Brain) ---
    # We use Google's Embedding model to convert text into mathematical vectors.
    status.info("🚀 Building Brain in batches...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectorstore = None
    batch_size = 5  # Small batch size to remain within API rate limits for large documents
    
    # This loop processes data in batches and includes a 'Retry Shield' (Exception Handling)
    # to prevent the application from crashing if the Google API is throttled.
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        
        success = False
        for attempt in range(3): # Exponential backoff retry logic
            try:
                if vectorstore is None:
                    # Initialize the FAISS vector database with the first batch
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    # Append subsequent batches to the existing knowledge base
                    vectorstore.add_documents(batch)
                success = True
                break  # Exit retry loop on success
            except Exception as e:
                if attempt < 2:
                    status.warning(f"🔄 API Busy. Retrying in 15s... (Attempt {attempt + 1})")
                    time.sleep(15) 
                else:
                    st.error("❌ Google API is heavily throttled. Please try again later.")
                    st.stop()
        
        if success:
            # Update the UI progress bar for a better user experience
            percent = int(((i + len(batch)) / len(docs)) * 100)
            status.warning(f"⏳ Syncing Knowledge Base: {percent}%")
            time.sleep(1) # Intentional delay to maintain API stability

    status.success("✅ Analysis Ready!")

    # --- STEP D: SETUP THE QA CHAIN ---
    # Create a Retrieval-QA chain using Gemini 1.5 Flash. 
    # This 'Retrieval' method ensures the AI only answers using the uploaded PDF data (Grounded AI).
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # --- STEP E: USER INTERACTION ---
    st.divider()
    user_q = st.text_input("Ask a question to compare your books (e.g., 'What are the different definitions of DAX?'):")
    if user_q:
        with st.spinner("Analyzing sources and synthesizing response..."):
            # The system retrieves relevant chunks from FAISS and generates an answer via Gemini.
            response = qa_chain.run(user_q)
            st.markdown(f"### 🤖 AI Synthesis\n{response}")

# 4. INITIAL PROMPT
elif not api_key:
    st.info("Please enter your Google API Key in the sidebar to begin.")
