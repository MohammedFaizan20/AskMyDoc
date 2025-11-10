
import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from apps.rag_pipeline import rag_pipeline
from apps.core.config import UPLOAD_DIR, APP_NAME, DESCRIPTION


UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title=APP_NAME,
    page_icon="📘",
    layout="wide",
)


st.markdown("""
    <style>
        .main {
            background-color: #f8fafc;
            padding: 1rem 3rem;
        }

        .stApp {
            max-width: 100%;
            margin: 0;
            padding: 0;
        }

        div[data-testid="stMarkdownContainer"] p {
            font-size: 16px !important;
            color: #1f2937;
        }

        .stButton>button {
            background-color: #2563eb;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-weight: 500;
            transition: 0.25s;
        }
        .stButton>button:hover {
            background-color: #1d4ed8;
            color: white;
            transform: scale(1.02);
        }

        h1, h2, h3 {
            color: #1e3a8a;
        }

        .answer-box {
            padding: 1em;
            background-color: #eef2ff;
            border-left: 5px solid #3b82f6;
            border-radius: 8px;
        }

        section[data-testid="stSidebar"] {
            background-color: #f1f5f9;
            border-right: 1px solid #e5e7eb;
        }

        /* Compact layout for full-screen fit */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 0rem !important;
        }


        div[data-testid="stFileUploaderDropzone"]::after {
            content: "Limit 50MB per file • Supported formats: PDF, TXT, MD";
            display: block;
            color: #6b7280;
            font-size: 0.85rem;
            text-align: center;
            margin-top: 6px;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown(f"<h1 style='text-align:center;'>📘 {APP_NAME}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; color:#374151;'>{DESCRIPTION}</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("ℹ️ About This App")
    st.write("""
    This RAG (Retrieval-Augmented Generation) demo lets you:

    1️⃣ Upload and embed your document  
    2️⃣ Ask questions using an LLM  
    3️⃣ Get context-aware answers instantly  
    """)
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + LangChain")

st.subheader("📤 Step 1: Upload & Ingest Document")

uploaded_file = st.file_uploader(
    "📤 Upload a `.pdf`, `.txt`, or `.md` file (Max size: 50MB)",
    type=["pdf", "txt", "md"],
    help="Limit 50MB per file • Supported formats: PDF, TXT, MD",
)

#Backend validation
if uploaded_file:
    if uploaded_file.size > 50 * 1024 * 1024:
        st.error("❌ File too large. Please upload a document under 50 MB.")
        st.stop()

    dest_path = UPLOAD_DIR / uploaded_file.name
    dest_path.write_bytes(uploaded_file.read())
    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

    if st.button("🚀 Ingest Document", use_container_width=True):
        with st.spinner("Embedding and storing document... ⏳"):
            try:
                rag_pipeline.ingest_file(dest_path)
                st.success(f"✅ '{uploaded_file.name}' ingested into vector DB!")
            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")

st.markdown("---")
st.subheader("💬 Step 2: Ask a Question")

query = st.text_area(
    "Ask something about your uploaded document:",
    placeholder="e.g. What is the main idea of this text?",
    height=80,
)

if st.button("🧠 Get Answer", use_container_width=True):
    if not query.strip():
        st.warning("⚠️ Please enter a question first.")
    else:
        with st.spinner("🤔 Thinking..."):
            try:
                answer = rag_pipeline.ask(query.strip())
                st.markdown(
                    f"""
                    <div class="answer-box">
                        <strong>🧠 Answer:</strong><br>{answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"❌ Query failed: {e}")
