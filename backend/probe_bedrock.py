import os, json, boto3
model_id = os.getenv("BEDROCK_CHAT_MODEL", "amazon.titan-text-express-v1")
region   = os.getenv("BEDROCK_REGION", "us-west-2")
profile  = os.getenv("AWS_PROFILE")

print("testing model:", model_id, "region:", region, "profile:", profile)

session = boto3.session.Session(profile_name=profile, region_name=region) if profile else boto3.session.Session(region_name=region)
brt = session.client("bedrock-runtime")

payload = {
    "inputText": "Say hello in one sentence.",
    "textGenerationConfig": {"maxTokenCount": 32, "temperature": 0.1, "topP": 0.9}
}

resp = brt.invoke_model(
    modelId=model_id,
    contentType="application/json",
    accept="application/json",
    body=json.dumps(payload),
)

print("status ok, raw:", resp["body"].read().decode("utf-8")[:400])
