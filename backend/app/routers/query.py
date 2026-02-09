# app/routers/query.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List
from sqlalchemy.orm import Session

from ..db import get_db
from ..services import embeddings, generator
from ..services import retriever  # <-- import the module above

router = APIRouter()

MAX_CHARS = 700
def _trim(txt: str, n: int = MAX_CHARS) -> str:
    return txt if len(txt) <= n else txt[:n].rsplit(" ", 1)[0] + "…"

# ------------ Models ------------

class QueryIn(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = Field(4, ge=1, le=20)

class Snippet(BaseModel):
    title: str
    seq: int
    text: str

class QueryOut(BaseModel):
    answer: str
    snippets: List[Snippet]

# ------------ Helpers ------------

def _fetch_top_chunks(db: Session, query_vec: List[float], top_k: int) -> List[Snippet]:
    """
    Returns a list of Snippet models (title, seq, text) chosen by nearest L2 distance.
    """
    rows = retriever.top_k_by_vector(db, query_vec, top_k)
    return [Snippet(title=r["title"], seq=r["seq"], text=_trim(r["text"])) for r in rows]

# ------------ Routes ------------

@router.post("/query", response_model=QueryOut)
def query(payload: QueryIn, db: Session = Depends(get_db)):
    # 1) Embed the question
    q_vecs = embeddings.embed_texts([payload.question])
    if not q_vecs or not q_vecs[0]:
        raise HTTPException(500, "Failed to embed query.")
    qvec = q_vecs[0]

    # 2) Retrieve top-k chunks
    snippets = _fetch_top_chunks(db, qvec, payload.top_k)
    if not snippets:
        return QueryOut(
            answer="I don’t have any embedded content yet to answer. Try ingesting a PDF first.",
            snippets=[]
        )

    # 3) Format passages for the LLM (title, text, seq)
    #    Keep it simple: generator expects list[tuple[title, text, idx]]
    passages = [(s.title, s.text, s.seq) for s in snippets]

    # 4) Call LLM
    try:
        ans = generator.generate_answer(payload.question, passages)
    except Exception as e:
        raise HTTPException(500, f"LLM generation failed: {e}")

    return QueryOut(answer=ans, snippets=snippets)


# useful for debugging retrieval without LLM
class SearchIn(BaseModel):
    query: str
    top_k: int = 4

@router.post("/search", response_model=List[Snippet])
def search_only(payload: SearchIn, db: Session = Depends(get_db)):
    q_vecs = embeddings.embed_texts([payload.query])
    if not q_vecs or not q_vecs[0]:
        raise HTTPException(500, "Failed to embed query.")
    return _fetch_top_chunks(db, q_vecs[0], payload.top_k)
