import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
import tempfile
st.set_page_config(page_title="Cross-Ref Engine", layout="wide")
# Sidebar for API Key and File Upload
with st.sidebar:
    st.title("⚙️ Setup")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    uploaded_files = st.file_uploader("Upload 2+ Textbooks (PDF)", type="pdf", accept_multiple_files=True)
st.title("📚 Academic Cross-Reference Engine")
st.write("Upload your textbooks to find contradictions or compare methods!")
if uploaded_files and api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
  
  # Process files
    all_pages = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            all_pages.extend(loader.load()) # Use .load() here for better control

    # NEW: Split the text into manageable slices (Chunking)
    #from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_pages)

    # Initialize Brain with chunks instead of full pages
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
# User Query
    user_q = st.text_input("What would you like to compare?")
    if user_q:
        response = qa_chain.run(user_q)
        st.markdown("### 🤖 Analysis")
        st.write(response)
elif not api_key:
    st.info("Please enter your OpenAI API key in the sidebar to begin.")
