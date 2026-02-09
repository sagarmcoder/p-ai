# P_AI - Enterprise Knowledge Retrieval System

Built to demonstrate backend engineering for enterprise RAG workflows using FastAPI, PostgreSQL/pgvector, AWS S3, and AWS Bedrock.

## Overview
P_AI ingests PDF documents, stores source files in S3, extracts and chunks text, generates vector embeddings, and serves retrieval-augmented answers through API endpoints.

The project is designed as a practical backend MVP for internal knowledge retrieval use cases.

## Core Features
- PDF ingestion pipeline (`/ingest/file`)
  - Upload PDF
  - Store original file in S3
  - Extract text (PyMuPDF with pdfminer fallback)
  - Clean OCR artifacts
  - Chunk text into retrieval units
  - Create embeddings via Bedrock Titan
  - Persist documents/chunks/embeddings in PostgreSQL (pgvector)
- Retrieval and answering
  - `POST /query/search`: vector retrieval only
  - `POST /query/query`: retrieval + Bedrock generation
- Configurable model/provider setup via environment variables
- Support for AWS profile-based local development

## Architecture
### Ingest Flow
1. Client uploads a PDF to `POST /ingest/file`
2. API stores bytes in S3
3. API extracts text from PDF and cleans OCR noise
4. API chunks text
5. API generates embeddings per chunk
6. API saves `documents`, `chunks`, and `embeddings`

### Query Flow
1. Client submits question to `POST /query/query`
2. API embeds the question
3. API retrieves top-k nearest chunks using pgvector distance (`<->`)
4. API builds a grounded prompt from snippets
5. API generates answer with Bedrock LLM
6. API returns answer + snippets

## Tech Stack
- FastAPI
- SQLAlchemy
- PostgreSQL + pgvector
- AWS S3
- AWS Bedrock (Titan embeddings + configurable chat model)
- PyMuPDF / pdfminer.six

## Repository Structure
- `backend/app/main.py` - FastAPI app setup and router registration
- `backend/app/routers/ingest.py` - ingestion API
- `backend/app/routers/query.py` - search/query APIs
- `backend/app/models.py` - SQLAlchemy models (`documents`, `chunks`, `embeddings`)
- `backend/app/services/embeddings.py` - Bedrock embedding service
- `backend/app/services/retriever.py` - pgvector nearest-neighbor retrieval
- `backend/app/services/generator.py` - Bedrock generation + prompting
- `backend/app/services/pdf.py` - PDF text extraction/cleaning
- `backend/app/services/chunker.py` - safe chunking logic
- `backend/app/services/s3.py` - S3 upload/read helpers
- `backend/.env.example` - environment template

## Prerequisites
- Python 3.11+
- PostgreSQL with `pgvector` extension enabled
- AWS credentials with access to:
  - S3 bucket
  - Bedrock runtime (embed + generation models)

## Environment Setup
Create and edit env file:

```bash
cd backend
cp .env.example .env
```

Set required values in `.env`:
- `DB_URL`
- `S3_BUCKET`
- `AWS_REGION`
- `BEDROCK_REGION`
- `BEDROCK_EMBED_MODEL`
- `BEDROCK_CHAT_MODEL`
- `AWS_PROFILE` (optional for local profile-based auth)

## Local Run
From project root:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload --port 8080
```

Health check:

```bash
curl http://127.0.0.1:8080/healthz
```

## API Endpoints
- `GET /healthz`
- `POST /ingest/file`
- `POST /query/search`
- `POST /query/query`

### Example: Ingest PDF
```bash
curl -X POST http://127.0.0.1:8080/ingest/file \
  -F "file=@sample.pdf"
```

### Example: Retrieval Only
```bash
curl -X POST http://127.0.0.1:8080/query/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Summarize the key findings","top_k":4}'
```

### Example: Retrieval + Answer
```bash
curl -X POST http://127.0.0.1:8080/query/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the main conclusions?","top_k":4}'
```

## Data Model
- `documents`: source metadata (`title`, `s3_uri`, `mime`)
- `chunks`: chunk text and sequence number
- `embeddings`: pgvector embedding per chunk

## Current Limitations
- No auth/multi-tenant isolation (MVP)
- No background queue for large ingest jobs
- Minimal automated testing in current repo
- Prompting/retrieval evaluation pipeline not yet formalized

## Future Plans
- Add sentence-transformers fallback for local/offline embedding workflows
- Harden AWS deployment (ECS/EC2 + RDS + IAM least privilege)
- Add async/background ingestion for large files
- Add evaluation suite for retrieval quality and answer grounding
- Add API auth, rate limits, and observability

## License
MIT License (see `LICENSE`).
