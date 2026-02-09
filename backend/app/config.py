# env vars (pydantic Settings)
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    DB_URL: str
    S3_BUCKET: str
    AWS_REGION: str = "ca-central-1"
    BEDROCK_REGION: str | None = None
    BEDROCK_EMBED_MODEL: str = "amazon.titan-embed-text-v2:0"
    BEDROCK_CHAT_MODEL: str = "amazon.titan-text-express-v1" # default

    # new: allow AWS profile to be set via .env
    AWS_PROFILE: str | None = None
    # add these
    DRY_INGEST: bool = False
    USE_PYMUPDF: bool = True
    EMBED_BATCH_SIZE: int = 64
    EMBED_TIMEOUT_SECS: int = 30

    # ✅ FIXED — use '=' not ':' and 'extra' not 'extras'
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

settings = Settings()


