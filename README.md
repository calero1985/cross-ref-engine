Project Overview and Technical Specifications
Purpose: The engine is designed as a synthesis tool to identify contradictions and compare methodologies across multiple academic documents.

Technical Stack: The application is built using a modern AI pipeline:

Frontend: Streamlit.

Orchestration: LangChain.

Vector Storage: FAISS.

Large Language Model: Google Gemini 1.5 Flash.

Reproducibility: Installation is standardized through a requirements.txt file to ensure the environment can be replicated across different platforms.

Core Features

RAG Architecture: Ensures AI responses are grounded and derived directly from the provided PDF sources.  


High-Speed Syncing: The system is optimized specifically for the Google Gemini 1.5 Flash model.  


Batch Processing: Specifically engineered to handle large textbook files without encountering API timeouts.  

Implementation and Operational Requirements
User Interface: Users must upload at least two PDF files and provide a valid Google API key to begin the analysis.

Resiliency: The system includes built-in retry logic (e.g., "Attempt 2") to handle instances where the Google API may be busy or heavily throttled.


Subscription Tier: A paid or prepaid tier for Google AI Studio is required when processing large PDF documents to avoid throttling errors.
