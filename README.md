📚 Academic Cross-Reference Engine
The Problem
Students and researchers are often overwhelmed by massive textbooks. Manually finding where authors contradict each other or where different books offer different methods is time-consuming and prone to human error.  

The Solution
We built a centralized "AI Brain" that reads thousands of pages across multiple documents simultaneously. It allows you to ask complex questions, such as: "How does Textbook A's definition of DAX differ from Textbook B's?".  

🧠 How It Works (The AI Pipeline)
The system uses a professional Retrieval-Augmented Generation (RAG) workflow to ensure accuracy:  


Ingestion: Extracts raw text from uploaded PDFs using PyPDFLoader.  


Chunking: Breaks text into 1000-character pieces with overlap so no context is lost.  


Vectorizing: Converts text into mathematical "embeddings" using Google’s models.  


Retrieval: When you ask a question, the system finds the most relevant text chunks in the FAISS vector database.  


Synthesis: Gemini 1.5 Flash analyzes the chunks and generates a grounded answer based only on your books.  

🚀 Key Innovations

The "Retry Shield": Built-in logic that automatically retries the connection (Attempt 1, 2, 3) if the API is busy or throttled.  


High-Speed Batching: Custom processing loops that handle large textbooks (8MB+) without crashing the system.  


Zero-Hallucination Mode: Because it uses RAG, the AI is strictly limited to the information in your PDFs.  

⚙️ Technical Stack

Frontend: Streamlit   


Orchestration: LangChain   


Database: FAISS   


LLM: Google Gemini 1.5 Flash   

🛠️ How to Run
Install requirements: pip install -r requirements.txt

Run the app: streamlit run app.py


Setup: Enter your Google API Key in the sidebar and upload at least two PDFs to start comparing!
