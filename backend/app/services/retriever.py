# retriever.py — pgvector search retriever
from typing import List, Dict, Any
from sqlalchemy import text
from sqlalchemy.orm import Session


def _to_pgvector_literal(vec: List[float]) -> str:
    # pgvector expects "[v1,v2,...]"
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def top_k_by_vector(db: Session, query_vec: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Returns dicts: {seq, title, text, distance}
    """
    vec_lit = _to_pgvector_literal(query_vec)

    sql = text("""
        SELECT DISTINCT ON (c.id)
            c.seq                                 AS seq,
            d.title                               AS title,
            c.text                                AS text,
            (e.vector <-> CAST(:query_vec AS vector)) AS distance
        FROM embeddings e
        JOIN chunks     c ON c.id = e.chunk_id
        JOIN documents  d ON d.id = c.document_id
        ORDER BY c.id, distance
        LIMIT :top_k
    """)

    rows = db.execute(sql, {"query_vec": vec_lit, "top_k": top_k}).mappings().all()

    return [
        {"seq": r["seq"], "title": r["title"], "text": r["text"], "distance": r["distance"]}
        for r in rows
    ]
