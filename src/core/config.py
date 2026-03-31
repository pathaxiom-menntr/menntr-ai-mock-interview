"""Application configuration using Pydantic settings."""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # Application
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["*"]  # Allow all origins

    # Database
    DATABASE_URL: str

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Azure OpenAI (Primary LLM)
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str = "2024-08-01-preview"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "AZURE_OPENAI_API-4.1"
    
    AZURE_SPEECH_KEY: str = ""
    AZURE_SPEECH_REGION: str = ""
    OPENAI_TTS_MODEL: str = "tts-1-hd"  # tts-1 or tts-1-hd (for text-to-speech) - using hd for more natural voice
    OPENAI_TTS_VOICE: str = "alloy"  # alloy, echo, fable, onyx, nova, shimmer

    # LiveKit
    LIVEKIT_API_KEY: str = ""
    LIVEKIT_API_SECRET: str = ""
    LIVEKIT_URL: str = ""
    LIVEKIT_WS_URL: str = ""  # WebSocket URL for agent connection (auto-detected from LIVEKIT_URL if not set)

    # File Uploads
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 10485760  # 10MB

    # Sandbox
    SANDBOX_TIMEOUT_SECONDS: int = 30
    SANDBOX_MEMORY_LIMIT: str = "128m"
    SANDBOX_CPU_LIMIT: str = "0.5"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignore extra environment variables
    )

    def get_azure_openai_client(self):
        """Get an AsyncAzureOpenAI client."""
        from openai import AsyncAzureOpenAI
        return AsyncAzureOpenAI(
            api_key=self.AZURE_OPENAI_API_KEY,
            azure_endpoint=self.AZURE_OPENAI_ENDPOINT,
            api_version=self.AZURE_OPENAI_API_VERSION,
        )


settings = Settings()


