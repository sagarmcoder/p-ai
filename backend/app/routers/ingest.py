
# app/routers/ingest.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import Document, Chunk, Embedding
from ..services import s3, pdf, chunker, embeddings
import os
import time,logging
from ..config import settings

logger = logging.getLogger("uvicorn.error")


router = APIRouter()

PDF_MIME = {"application/pdf"}
MAX_CHUNKS = 4_000  # guardrail to avoid runaway inserts on weird PDFs


@router.post("/file")
async def ingest_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    t0 = time.time()
    if file.content_type not in ("application/pdf",):
        raise HTTPException(400, "Only PDF supported for MVP.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    logger.info(f"[ingest] received {file.filename} size={len(data)} bytes")

    # 1) S3
    s3_uri = s3.put_bytes(file.filename, data, file.content_type)
    logger.info(f"[ingest] S3 put ok uri={s3_uri} dt={time.time()-t0:.3f}s")

    # 2) PDF text
    t1 = time.time()
    try:
        text = pdf.extract_text(data)
    except Exception as e:
        raise HTTPException(400, f"PDF parse failed: {e}")
    logger.info(f"[ingest] pdf.extract_text ok chars={len(text)} dt={time.time()-t1:.3f}s")

    # 3) chunk
    # in app/routers/ingest.py, right before chunking
    t2 = time.time()
    logger.info("[ingest] starting chunking")
    try:
        chunks = list(chunker.safe_chunks(text))
        # Fallback if your chunker returned nothing or crashes in edge cases:
        # if not chunks:
        #     size = 1200  # char-based fallback
        #     chunks = [text[i:i+size] for i in range(0, len(text), size)]

    except Exception as e:
        logger.exception("[ingest] chunking failed")
        raise HTTPException(status_code=500, detail=f"Chunking failed: {e}")
    logger.info(f"[ingest] chunked -> {len(chunks)} chunks dt={time.time()-t2:.3f}s")


    # DRY PATH (skip embeddings entirely)
    if settings.DRY_INGEST:
        t3 = time.time()
        try:
            doc = Document(title=file.filename, s3_uri=s3_uri, mime=file.content_type)
            db.add(doc); db.flush()
            to_add = [Chunk(document_id=doc.id, seq=i, text=t, meta_json={}) for i,t in enumerate(chunks)]
            db.bulk_save_objects(to_add, return_defaults=True)
            db.commit()
        except Exception:
            db.rollback()
            raise
        logger.info(f"[ingest] DRY persisted chunks={len(to_add)} dt={time.time()-t3:.3f}s")
        return {"document_id": doc.id, "chunks": len(to_add), "note": "DRY_INGEST"}

    # 4) embeddings (non-dry)
    t4 = time.time()
    vecs = embeddings.embed_texts(chunks)
    logger.info(f"[ingest] embeddings ok vecs={len(vecs)} dt={time.time()-t4:.3f}s")

    # 5) persist in bulk
    t5 = time.time()
    doc = Document(title=file.filename, s3_uri=s3_uri, mime=file.content_type)
    db.add(doc); db.flush()
    chunk_rows = [Chunk(document_id=doc.id, seq=i, text=t, meta_json={}) for i,t in enumerate(chunks)]
    db.bulk_save_objects(chunk_rows, return_defaults=True); db.flush()
    emb_rows = [Embedding(chunk_id=c.id, vector=v) for c, v in zip(chunk_rows, vecs)]
    db.bulk_save_objects(emb_rows)
    db.commit()
    logger.info(f"[ingest] db bulk persisted dt={time.time()-t5:.3f}s total={time.time()-t0:.3f}s")

    return {"document_id": doc.id, "chunks": len(chunk_rows)}

