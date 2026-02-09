#bedrock adapter (titan v2) for embeddings

# app/services/embeddings.py
# Bedrock Titan V2 embeddings (batched, with timeouts & retries)

import os, json, boto3, time
from botocore.config import Config
from ..config import settings

# region & tuning from env
_REGION   = settings.BEDROCK_REGION or settings.AWS_REGION
_AWS_PROFILE = settings.AWS_PROFILE
_BATCH    = int(os.getenv("EMBED_BATCH_SIZE", "64"))
_TIMEOUT  = int(os.getenv("EMBED_TIMEOUT_SECS", "30"))

# 1. Create a Boto3 Session with the specific profile
# If _AWS_PROFILE is None or empty, the Session will fall back to default credentials.
session = boto3.Session(profile_name=_AWS_PROFILE, region_name=_REGION)
# configurable throttle (ms) to be gentle with the API
_MS_BETWEEN_CALLS = int((getattr(settings, "EMBED_THROTTLE_MS", None) or 50))
# 2. Create the Bedrock client from that Session object
# Note: region_name can be passed here or in the Session creation.
bedrock = session.client(
    "bedrock-runtime",
    config=Config(
        retries={"max_attempts": 5, "mode": "standard"},
        read_timeout=_TIMEOUT,
        connect_timeout=_TIMEOUT,
    ),
)

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def _embed_one(text: str) -> list[float]:
    body = {"inputText": text}
    resp = bedrock.invoke_model(
        modelId=settings.BEDROCK_EMBED_MODEL,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    data = json.loads(resp["body"].read())
    # Titan v2 returns single vector under "embedding"
    if "embedding" in data:
        return data["embedding"]
    # Fallback (defensive)
    if "embeddings" in data and data["embeddings"]:
        return data["embeddings"][0]
    raise RuntimeError(f"Unexpected embed response keys={list(data.keys())}")

def embed_texts(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i, t in enumerate(texts):
        vec = _embed_one(t)
        out.append(vec)
        # light throttle to avoid bursts / throttling
        if _MS_BETWEEN_CALLS:
            time.sleep(_MS_BETWEEN_CALLS / 1000.0)
    return out

'''
def _call_bedrock(batch: list[str]) -> list[list[float]]:
    # Titan v2 embed typically accepts an array in `inputText`
    # If your model card shows a different schema, adapt here.
    payload = {"inputText": batch}
    resp = bedrock.invoke_model(
        modelId=settings.BEDROCK_EMBED_MODEL,
        body=json.dumps(payload),
        accept="application/json",
        contentType="application/json",
    )
    data = json.loads(resp["body"].read())

    # Handle single vs batch keys defensively
    if "embeddings" in data:
        return data["embeddings"]
    if "embedding" in data:
        return [data["embedding"]]

    raise RuntimeError(f"Unexpected embed response keys: {list(data.keys())}")

def embed_texts(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for block in _chunks(texts, _BATCH):
        out.extend(_call_bedrock(block))
    return out
'''


