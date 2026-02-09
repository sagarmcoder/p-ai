# Simple text chunking (splits text into overlapping chunks)
from __future__ import annotations
import os
from typing import Iterable


def simple_chunks(text: str, max_chars: int = 1500, overlap: int = 200):
    i, n = 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        yield text[i:j]
        i = j - overlap
        if i < 0: i = 0


# app/services/chunker.py


# Tunables via env (safe defaults)
TARGET = int(os.getenv("CHUNK_TARGET_CHARS", "1200"))
OVERLAP = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))
MAX_CHUNKS = int(os.getenv("CHUNK_MAX_CHUNKS", "5000"))
MAX_TEXT = int(os.getenv("CHUNK_MAX_TEXT_CHARS", "5_000_000"))  # hard cap

def _normalize(s: str) -> str:
    # cheap normalization: strip nulls, collapse obvious whitespace runs
    s = s.replace("\x00", "")
    # Avoid costly regex: simple replaces
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # limit abyssal inputs
    return s[:MAX_TEXT]

def safe_chunks(text: str,
                target: int = TARGET,
                overlap: int = OVERLAP) -> Iterable[str]:
    """
    Paragraph-first, char-window fallback splitter.
    No regex, no tokenizers, no downloads, no multiprocessing.
    Yields at most MAX_CHUNKS chunks.
    """
    t = _normalize(text)
    if not t:
        return

    # First split on blank lines to respect paragraphs
    paragraphs = t.split("\n\n")

    cur = []
    cur_len = 0
    produced = 0

    def flush():
        nonlocal cur, cur_len, produced
        if cur:
            if produced >= MAX_CHUNKS:
                return
            chunk = "\n\n".join(cur).strip()
            if chunk:
                produced += 1
                yield chunk
            cur = []
            cur_len = 0

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if cur_len + len(p) + (2 if cur else 0) <= target:
            cur.append(p)
            cur_len += len(p) + (2 if cur_len > 0 else 0)
        else:
            # emit current
            for out in flush() or []:
                yield out
            # start new window; if a single paragraph is huge, hard-slice it
            if len(p) <= target:
                cur = [p]
                cur_len = len(p)
            else:
                start = 0
                while start < len(p) and produced < MAX_CHUNKS:
                    end = start + target
                    yield p[start:end]
                    produced += 1
                    # simple overlap between slices of a huge paragraph
                    start = max(end - overlap, start + 1)

    for out in flush() or []:
        yield out

    # If we still have some tail and haven't yielded it:
    # (handled by flush above)
    return
