# AskMyDoc - AI-Powered Document Question Answering Application

## Overview

**AskMyDoc** is an AI-powered document question-answering system that enables users to upload `.pdf`, `.txt`, or `.md` files and query their contents in natural language.  
The application leverages a **Retrieval-Augmented Generation (RAG)** architecture built using **LangChain**, **FAISS**, and **Hugging Face Inference API**, providing precise, context-aware answers.

The system is implemented with **Python** and **Streamlit** for a simple and interactive user experience and deployed on **Hugging Face Spaces**.

🔗 **Live Demo:** [AskMyDoc on Hugging Face Spaces](https://huggingface.co/spaces/faizan20/askmydoc)

---

## Key Features

- Upload and analyze `.pdf`, `.txt`, or `.md` documents (up to 50 MB)
- Contextual question answering using RAG pipeline
- Real-time text embedding and retrieval using **FAISS**
- **Streamlit**-based user interface
- Secure integration with **Hugging Face Inference API**
- Hidden token management using `.env` and repository secrets
- Lightweight, modular, and fully deployable architecture

---

## Models and Components

| Component  |  Model | Description |
|------------|--------|-------------|
| **Language Model (LLM)** | [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | Instruction-tuned large language model used for generating answers based on retrieved context |
| **Embedding Model**      | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Sentence Transformer model for semantic text embedding |
| **Vector Store**         | FAISS | Efficient vector similarity search for document retrieval |

---

## Technology Stack

**Programming Language:** Python 3.10+  
**Frontend Framework:** Streamlit  
**Core Framework:** LangChain  
**LLM Provider:** Hugging Face Inference API  
**Embeddings:** Sentence Transformers  
**Vector Database:** FAISS  
**Deployment Platform:** Hugging Face Spaces  
**Supporting Libraries:** Pydantic, Python-dotenv, PyPDF2, Pandas, NumPy, TQDM  

---

## Folder Structure

askmydoc/
├── apps/
│   ├── streamlit_app.py         # Main Streamlit frontend
│   ├── rag_pipeline.py          # Core RAG pipeline implementation
│   └── core/
│       ├── config.py            # Environment variables and configuration
│       ├── utils.py             # Document parsing and text extraction utilities
│
├── requirements.txt             # Dependencies
├── .huggingface.yml             # Hugging Face Space configuration
├── runtime.txt                  # Python runtime version
└── README.md                    # Project documentation


## Functional Workflow

## Document Upload
The user uploads a .pdf, .txt, or .md file via the Streamlit interface.

## Text Extraction
Text is extracted using PyPDF2 or plain-text parsing utilities.

## Text Chunking
Extracted text is segmented into smaller, context-preserving chunks using RecursiveCharacterTextSplitter.

## Embedding Generation
Each chunk is transformed into a dense vector using the sentence-transformers/all-MiniLM-L6-v2 embedding model.

## Vector Indexing
Generated embeddings are stored in a FAISS vector database for efficient similarity-based retrieval.

## Query Handling
The user submits a question, which is embedded using the same model, and the most relevant text chunks are retrieved from FAISS.

## Answer Generation
The retrieved context is passed to the Mistral-7B-Instruct model through the Hugging Face Inference API, which produces the final context-aware answer.

## Local Setup
1. Clone the Repository
git clone https://github.com/MohammedFaizan20/askmydoc.git
cd askmydoc

2. Create and Activate a Virtual Environment
python -m venv venv
venv\Scripts\activate    # For Windows
# or
source venv/bin/activate # For macOS/Linux

3. Install Dependencies
pip install -r requirements.txt

4. Add Environment Variables
Create a .env file in the project root:
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

5. Run the Application
streamlit run apps/streamlit_app.py

## Deployment on Hugging Face Spaces

1.Push the project to your Hugging Face Space repository.
2.Open Settings → Repository Secrets and add: HUGGINGFACEHUB_API_TOKEN
3.Save and rebuild the Space.
4.The Streamlit app will automatically deploy and become accessible at the Space URL

## Security and Configuration

1.Sensitive credentials are securely managed using .env and Hugging Face Secrets.
2..env and local environment directories are excluded via .gitignore.
3.No API tokens or secrets are exposed in the public repository.
4.All dependencies are version-locked for stable builds.

## Future Enhancements

1.Multi-document ingestion and cross-document querying
2.Persistent FAISS storage (for long-term embeddings)
3.Support for DOCX and CSV formats
4.Enhanced conversational interface for multi-turn dialogue
5.Integration with alternative LLMs (e.g., Llama 3, Mixtral, Claude)
