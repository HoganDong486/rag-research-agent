from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from rag import RAGPipeline
from agent import RAGAgent
from context_loader import ProgressiveLoader
from self_improver import SelfImprover

app = FastAPI(title="RAG Research Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()
agent = RAGAgent(rag)
loader = ProgressiveLoader()
improver = SelfImprover()

query_counter = 0


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    use_progressive: Optional[bool] = True


class AgentQueryRequest(BaseModel):
    task: str
    max_iterations: Optional[int] = 3
    use_progressive: Optional[bool] = True


class FeedbackRequest(BaseModel):
    query_index: int
    rating: int  # 1-5


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    contents = await file.read()
    num_chunks = rag.ingest_pdf(contents, file.filename)
    return {"filename": file.filename, "chunks_indexed": num_chunks}


@app.post("/query")
def query(req: QueryRequest):
    global query_counter
    result = rag.query(req.question, top_k=req.top_k)

    retrieval_score = sum(s["score"] for s in result["sources"]) / max(len(result["sources"]), 1)
    improver.record_query(
        query=req.question,
        answer=result["answer"],
        source_count=len(result["sources"]),
        retrieval_score=retrieval_score,
    )
    query_counter += 1

    if req.use_progressive:
        result["context_layers"] = loader.get_efficiency_report()

    return result


@app.post("/agent/query")
def agent_query(req: AgentQueryRequest):
    result = agent.execute(req.task, max_iterations=req.max_iterations)

    improver.record_query(
        query=req.task,
        answer=result["synthesis"],
        source_count=result["sources_consulted"],
        retrieval_score=1.0,
    )
    return result


@app.get("/documents")
def list_documents():
    docs = rag.list_documents()
    return {"documents": docs, "count": len(docs)}


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    improver.record_feedback(req.query_index, req.rating)
    return {"status": "feedback_recorded", "rating": req.rating}


@app.get("/improve")
def get_improvement_suggestions():
    return improver.suggest_improvements(rag)


@app.get("/history")
def get_query_history(limit: int = 10):
    return {"history": improver.get_history(limit), "total": len(improver.history)}


@app.get("/stats")
def get_stats():
    return {
        "documents_indexed": rag.list_documents(),
        "total_chunks": rag.collection.count(),
        "queries_served": query_counter,
        "efficiency": loader.get_efficiency_report(),
        "improvement": improver.suggest_improvements(rag),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
