# RAG Research Agent

LLM-powered research paper retrieval and Q&A system with autonomous agent mode.

## Architecture

```
User Query -> [Agent Planner] -> [Multi-Query Retrieval] -> [RAG Pipeline] -> [Synthesize] -> Answer
                                    |
                              ChromaDB Vector Store (research papers)
```

## Features

- **PDF Ingestion**: Upload academic papers, auto-extract text, chunk, embed
- **RAG Q&A**: Ask questions, get cited answers with source tracking
- **Autonomous Agent Mode**: Agent breaks down complex questions, runs multi-query search, synthesizes findings
- **Real-time Source Attribution**: Every answer linked to specific paper sections

## Quick Start

```bash
cd backend
pip install -r requirements.txt
export OPENAI_API_KEY=sk-your-key-here
python main.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload PDF for indexing |
| POST | `/query` | RAG-enhanced question answering |
| POST | `/agent/query` | Autonomous multi-query research agent |
| GET | `/documents` | List indexed documents |

## Example

```bash
# Upload a paper
curl -X POST http://localhost:8000/upload -F "file=@paper.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main contribution of this paper?"}'

# Agent mode
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"task": "Compare the model architectures used in the uploaded papers and identify the best performing approach"}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Custom API endpoint |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Vector DB storage |
| `CHUNK_SIZE` | `1000` | Text chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |

## Tech Stack

- **Backend**: Python · FastAPI · ChromaDB · OpenAI API
- **Vector Store**: ChromaDB with cosine similarity
- **Agent**: Multi-query planner + synthesis pipeline
