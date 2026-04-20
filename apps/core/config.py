import os
from pathlib import Path
from dotenv import load_dotenv

#PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_STORE_PATH = DATA_DIR / "vector_store"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

#MODEL CONFIG
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not loaded — model inference will fail.")
else:
    print("Google API key loaded successfully.")

#APP META
APP_NAME = "AskMyDocs"
DESCRIPTION = "Upload your files. Ask anything. Get precise, AI-powered answers with Retrieval-Augmented Generation (RAG) — built using LangChain and Streamlit."
VERSION = "1.0.0"
