import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
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
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Process files
    all_pages = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            all_pages.extend(loader.load_and_split())
    
    # Initialize Brain
    vectorstore = FAISS.from_documents(all_pages, OpenAIEmbeddings())
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o"),
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
