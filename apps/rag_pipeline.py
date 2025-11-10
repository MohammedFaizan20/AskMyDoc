from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from apps.core.config import EMBEDDING_MODEL, LLM_MODEL, HF_TOKEN
from apps.core.utils import extract_text
from huggingface_hub import InferenceClient
from langchain.llms.base import LLM
from pydantic.v1 import PrivateAttr
from typing import Optional, List

class HFInferenceLLM(LLM):

    model_id: str
    token: str
    temperature: float = 0.3
    max_new_tokens: int = 512
    top_p: float = 0.9

    _client: Optional[InferenceClient] = PrivateAttr()

    def __init__(self, model_id: str, token: str, temperature: float = 0.3, max_new_tokens: int = 512, top_p: float = 0.9):
        super().__init__(
            model_id=model_id,
            token=token,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
        )
        self._client = InferenceClient(model=model_id, token=token)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            resp = self._client.chat_completion(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=False,
            )
            return resp.choices[0].message["content"].strip()
        except Exception as e:
            return f"HF Inference Error: {e}"

    @property
    def _identifying_params(self):
        return {
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"


class RAGPipeline:
    def __init__(self):
        if not HF_TOKEN:
            raise RuntimeError("Missing HUGGINGFACEHUB_API_TOKEN in .env")

        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = None


        self.llm = HFInferenceLLM(
            model_id=LLM_MODEL,
            token=HF_TOKEN,
            temperature=0.3,
            max_new_tokens=512,
        )

    def ingest_file(self, file_path: Path):
        text = extract_text(file_path)
        if not text:
            raise ValueError("The document is empty or unreadable.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_text(text)

        self.vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            metadatas=[{"source": file_path.name}] * len(chunks),
        )

    def ask(self, query: str) -> str:
        if not self.vectorstore:
            return "Please ingest a document first."

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
        )

        result = qa_chain.invoke({"query": query})
        return result.get("result", "No answer found.")


rag_pipeline = RAGPipeline()
