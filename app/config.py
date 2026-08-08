from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    LANGSMITH_TRACING = os.getenv(
        "LANGSMITH_TRACING",
        "false"
    )

    LANGSMITH_ENDPOINT = os.getenv(
        "LANGSMITH_ENDPOINT"
    )

    LANGSMITH_API_KEY = os.getenv(
        "LANGSMITH_API_KEY"
    )

    LANGSMITH_PROJECT = os.getenv(
        "LANGSMITH_PROJECT",
        "InvestAI"
    )
    
    MODEL_NAME = "llama-3.1-8b-instant"
    EVALUATION_MODEL = "llama-3.3-70b-versatile"
    DOCUMENTS_PATH = "documents"
    VECTOR_DB_PATH = "chroma_db"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 500

    CHUNK_OVERLAP = 100

    TOP_K_RESULTS = 3