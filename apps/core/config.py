import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = Path("D:/pythonProject7/.env")

load_dotenv(dotenv_path=ENV_PATH)

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_STORE_PATH = DATA_DIR / "vector_store"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# === MODEL CONFIG ===
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


if not HF_TOKEN:
    print("Token not loaded — model inference will fail.")
else:
    print("Hugging Face token loaded successfully.")


APP_NAME = "AskMyDocs"
DESCRIPTION = "Upload your files. Ask anything. Get precise, AI-powered answers with Retrieval-Augmented Generation (RAG) — built using LangChain and Streamlit."
VERSION = "1.0.0"
