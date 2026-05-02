import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Cross-Ref Engine", layout="wide")

with st.sidebar:
    st.title("⚙️ Setup")
    api_key = st.text_input("Enter Google API Key", type="password")
    uploaded_files = st.file_uploader("Upload Textbooks (PDF)", type="pdf", accept_multiple_files=True)

st.title("📚 Academic Cross-Reference Engine")

if uploaded_files and api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    status = st.empty()
    
    # Step A: Load PDFs
    all_pages = []
    status.info("📖 Reading documents...")
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            all_pages.extend(loader.load())

    # Step B: Chunking
    status.info("✂️ Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_pages)

    # Step C: High-Speed Syncing (You have $25 credit now!)
    status.info("🚀 Building Brain (High-Speed Mode)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # No more loops, no more sleep timers!
    vectorstore = FAISS.from_documents(docs, embeddings)
    status.success("✅ Analysis Ready!")

    # Step D: Setup QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # Step E: User Query
    st.divider()
    user_q = st.text_input("Ask a question about your books:")
    if user_q:
        with st.spinner("Analyzing..."):
            response = qa_chain.run(user_q)
            st.markdown(f"### 🤖 AI Analysis\n{response}")

elif not api_key:
    st.info("Please enter your Google API Key to begin.")
