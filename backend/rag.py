import os
import io
import hashlib
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from openai import OpenAI


class RAGPipeline:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        self.collection_name = "research_papers"
        self.chunk_size = int(os.environ.get("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "200"))
        self.embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.llm_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_data")
        self.chroma = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def extract_text(self, pdf_bytes: bytes) -> str:
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        return "\n".join(texts)

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [d.embedding for d in response.data]

    def ingest_pdf(self, pdf_bytes: bytes, filename: str) -> int:
        text = self.extract_text(pdf_bytes)
        if not text.strip():
            return 0
        chunks = self.chunk_text(text)
        embeddings = self.embed_texts(chunks)

        doc_id = hashlib.md5(pdf_bytes).hexdigest()[:12]
        self.collection.delete(ids=self.collection.get(where={"source": filename}).get("ids", []))

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": filename, "chunk_index": i, "chunk_count": len(chunks)}
            for i in range(len(chunks))
        ]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.embed_texts([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                docs.append({
                    "content": results["documents"][0][i],
                    "source": results["metadatas"][0][i].get("source", "unknown"),
                    "score": 1.0 - (results["distances"][0][i] if results["distances"] else 0),
                    "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                })
        return docs

    def generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        context = "\n\n---\n\n".join(
            [f"[Source: {c['source']}, Chunk {c['chunk_index']}]\n{c['content']}" for c in context_chunks]
        )
        system_prompt = (
            "You are a research assistant specializing in academic papers. "
            "Answer questions based on the provided context. "
            "Cite specific sources when possible. "
            "If the context does not contain enough information, say so clearly."
        )
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def query(self, question: str, top_k: int = 5) -> Dict:
        chunks = self.retrieve(question, top_k=top_k)
        if not chunks:
            return {
                "question": question,
                "answer": "No relevant documents found. Please upload some papers first.",
                "sources": [],
            }
        answer = self.generate_answer(question, chunks)
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"source": c["source"], "score": round(c["score"], 4), "snippet": c["content"][:200] + "..."}
                for c in chunks
            ],
        }

    def list_documents(self) -> List[str]:
        results = self.collection.get(include=["metadatas"])
        if results["metadatas"]:
            sources = list(set(m["source"] for m in results["metadatas"] if m))
            return sorted(sources)
        return []
