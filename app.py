import streamlit as st
import os
import tempfile
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Page Configuration
st.set_page_config(page_title="Cross-Ref Engine", layout="wide")

# 2. Sidebar Setup
with st.sidebar:
    st.title("⚙️ Setup")
    api_key = st.text_input("Enter Google API Key", type="password")
    uploaded_files = st.file_uploader("Upload 2+ Textbooks (PDF)", type="pdf", accept_multiple_files=True)

st.title("📚 Academic Cross-Reference Engine")
st.write("Upload your textbooks to find contradictions or compare methods!")

# 3. Main Logic (Runs only when files and key are present)
if uploaded_files and api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Progress Bar UI
    status_text = st.empty()
    
    # Step A: Load all PDF pages into one list
    all_pages = []
    status_text.info("Reading PDFs...")
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            all_pages.extend(loader.load())

    # Step B: Split text into small chunks
    status_text.info("Splitting text into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(all_pages)

    # Step C: Create Brain (Embeddings) with Batching to avoid Rate Limits
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    batch_size = 5  
    vectorstore = None
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        
        # Update progress
        percent = int((i / len(docs)) * 100)
        status_text.warning(f"⏳ Syncing knowledge... {percent}% complete")
        time.sleep(10) # Breathing room for the API

    status_text.success("✅ Knowledge Base Ready!")

    # Step D: Setup the Question/Answer Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # Step E: User Interaction
    st.divider()
    user_q = st.text_input("What would you like to compare or find in these books?")
    if user_q:
        with st.spinner("Analyzing sources..."):
            response = qa_chain.run(user_q)
            st.markdown("### 🤖 AI Analysis")
            st.write(response)

# 4. Prompt for API Key if missing
elif not api_key:
    st.info("Please enter your Google API Key in the sidebar to begin.")
