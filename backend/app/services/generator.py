# Bedrock LLM call + prompt
import json
import logging
from typing import List, Tuple

import boto3
from botocore.exceptions import ClientError

# pull settings from your pydantic Settings
from ..config import settings

logger = logging.getLogger(__name__)

# ---- Config (prefer BEDROCK_REGION over AWS_REGION) ----
BEDROCK_REGION = settings.BEDROCK_REGION or settings.AWS_REGION or "us-west-2"
MODEL_ID = settings.BEDROCK_CHAT_MODEL  # e.g., "amazon.titan-text-express-v1" or Claude/Llama id

# Honor AWS profile if provided
if settings.AWS_PROFILE:
    _session = boto3.session.Session(profile_name=settings.AWS_PROFILE, region_name=BEDROCK_REGION)
else:
    _session = boto3.session.Session(region_name=BEDROCK_REGION)

bedrock = _session.client("bedrock-runtime")


'''# -------------------- RAG Prompt Builder -------------------- #

def _format_passages(passages: List[Tuple[str, str, int]]) -> str:
    # passages: [(title, text, idx), ...]
    lines = []
    for title, text, idx in passages:
        lines.append(f"### [{idx}] {title}\n{text}")
    return "\n\n".join(lines)


def _build_rag_prompt(question: str, passages: List[Tuple[str, str, int]]) -> str:
    context = _format_passages(passages)
    prompt = (
        "You are a helpful assistant. Use ONLY the provided context to answer.\n"
        "If the answer is not in the context, say \"I don’t know based on the provided documents.\" "
        "Cite snippets by their [index] where relevant.\n\n"
        f"Question:\n{question}\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Answer:"
    )
    return prompt
'''


# --- replace your _build_rag_prompt with this pair --- #

def _format_passages(passages: List[Tuple[str, str, int]]) -> str:
    lines = []
    for title, text, idx in passages:
        # keep context clean and compact
        lines.append(f"[{idx}] {title}\n{text}")
    return "\n\n".join(lines)


def _build_task_prompt(question: str, passages: List[Tuple[str, str, int]]) -> str:
    """
    Build a task-aware prompt. If the user asks to 'summarize', use a
    stricter summarization instruction that discourages verbatim copying.
    Otherwise use a general RAG QA instruction.
    """
    context = _format_passages(passages)
    q_lower = question.strip().lower()

    if "summarize" in q_lower or "summary" in q_lower:
        return (
            "You are a helpful assistant. Summarize the content strictly from the provided context.\n"
            "Rules:\n"
            " • Write a concise, paraphrased summary in your own words (no more than 90–120 words).\n"
            " • Do NOT copy sentences verbatim; avoid using more than 5 consecutive words from the context.\n"
            " • Cover only the most important ideas; remove repetition and filler.\n"
            " • If useful, cite snippet indices like [0], [1] when attributing key points.\n\n"
            f"Context:\n{context}\n\n"
            f"Task: Summarize the above context for the question: “{question}”.\n"
            "Answer:"
        )

    # default RAG QA prompt
    return (
        "You are a helpful assistant. Use ONLY the provided context to answer.\n"
        "If the answer is not in the context, say: “I don’t know based on the provided documents.”\n"
        "Cite snippets by their [index] where relevant.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


# -------------------- Payload builders per model family -------------------- #

def _payload_for_titan(prompt: str) -> dict:
    # Titan Text (G1 Express / Express v1 / Lite v1)
    return {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.4, # adjust as needed was 0.2
            "topP": 0.9,
        },
    }


def _payload_for_claude(prompt: str) -> dict:
    # Anthropic Messages API (Bedrock)
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.2,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    }


def _payload_for_llama(prompt: str) -> dict:
    # Meta Llama Instruct (Bedrock)
    return {
        "prompt": f"[INST] {prompt} [/INST]",
        "max_gen_len": 512,
        "temperature": 0.2,
        "top_p": 0.9,
    }


def _build_payload(model_id: str, prompt: str) -> dict:
    mid = model_id.lower()
    if mid.startswith("amazon.titan-text-g1") or mid.startswith("amazon.titan-text-express") or mid.startswith("amazon.titan-text-lite") or "titan" in mid:
        return _payload_for_titan(prompt)
    if mid.startswith("anthropic.claude"):
        return _payload_for_claude(prompt)
    if mid.startswith("meta.llama"):
        return _payload_for_llama(prompt)
    raise ValueError(f"Unsupported BEDROCK_MODEL_ID: {model_id}")


# -------------------- Response parsers per model family -------------------- #

def _parse_titan(body: dict) -> str:
    # Titan may return "outputText" top-level OR "results"[0]["outputText"]
    if isinstance(body, dict):
        if isinstance(body.get("outputText"), str):
            return body["outputText"]
        results = body.get("results") or []
        if results and isinstance(results[0], dict):
            ot = results[0].get("outputText")
            if isinstance(ot, str):
                return ot
        # Some variants:
        if "generations" in body and body["generations"]:
            g0 = body["generations"][0]
            if isinstance(g0, dict):
                return g0.get("text", "") or g0.get("generation", "")
    return ""


def _parse_claude(body: dict) -> str:
    parts = body.get("content", [])
    out = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            out.append(p.get("text", ""))
    return "".join(out)


def _parse_llama(body: dict) -> str:
    if "generation" in body:
        return body["generation"]
    if "generations" in body and body["generations"]:
        g0 = body["generations"][0]
        if isinstance(g0, dict):
            return g0.get("text", "") or g0.get("generation", "")
    if "outputs" in body and body["outputs"]:
        return body["outputs"][0].get("text", "")
    return ""


def _parse_response(model_id: str, raw_body: bytes) -> str:
    try:
        body = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return raw_body.decode("utf-8", errors="ignore")

    mid = model_id.lower()
    if mid.startswith("amazon.titan-text-g1") or mid.startswith("amazon.titan-text-express") or mid.startswith("amazon.titan-text-lite") or "titan" in mid:
        return _parse_titan(body)
    if mid.startswith("anthropic.claude"):
        return _parse_claude(body)
    if mid.startswith("meta.llama"):
        return _parse_llama(body)
    # fallback
    return body.get("outputText") or body.get("generation") or ""


# -------------------- Public API -------------------- #

def generate_answer(question: str, passages: List[Tuple[str, str, int]]) -> str:
    """
    Build a RAG prompt from passages and call Bedrock.
    passages: [(title, text, idx), ...]
    """
    prompt = _build_task_prompt(question, passages)
    payload = _build_payload(MODEL_ID, prompt)
    # Debug: model + payload keys (safe)
    logger.debug("Invoking model=%s in region=%s with keys=%s", MODEL_ID, BEDROCK_REGION, list(payload.keys()))

    
    #logger.debug("Invoking model=%s in region=%s with keys=%s", MODEL_ID, BEDROCK_REGION, list(payload.keys()))

    logger.error("BEDROCK effective -> region=%s model=%s", BEDROCK_REGION, MODEL_ID)

    try:
        resp = bedrock.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
    except ClientError as e:
        msg = e.response.get("Error", {}).get("Message", str(e))
        # surface exact Bedrock error (your FastAPI will wrap into 500)
        raise RuntimeError(msg)

    raw = resp.get("body")
    raw_bytes = raw.read() if hasattr(raw, "read") else (raw or b"")

    text = _parse_response(MODEL_ID, raw_bytes) or ""
    if not text.strip():
        snippet = raw_bytes[:300].decode("utf-8", errors="ignore")
        raise RuntimeError(f"Empty LLM response. Body snippet: {snippet}")
    return text.strip()
