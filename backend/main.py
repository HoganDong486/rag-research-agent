from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from rag import RAGPipeline
from agent import RAGAgent

app = FastAPI(title="RAG Research Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()
agent = RAGAgent(rag)


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class AgentQueryRequest(BaseModel):
    task: str
    max_iterations: Optional[int] = 3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    contents = await file.read()
    num_chunks = rag.ingest_pdf(contents, file.filename)
    return {"filename": file.filename, "chunks_indexed": num_chunks}


@app.post("/query")
def query(req: QueryRequest):
    result = rag.query(req.question, top_k=req.top_k)
    return result


@app.post("/agent/query")
def agent_query(req: AgentQueryRequest):
    result = agent.execute(req.task, max_iterations=req.max_iterations)
    return result


@app.get("/documents")
def list_documents():
    docs = rag.list_documents()
    return {"documents": docs, "count": len(docs)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
