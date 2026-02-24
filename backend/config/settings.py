"""
Application settings and configuration management
Production-ready configuration with environment variable support
"""

import os
from typing import Optional, List
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings with validation"""

    # -------------------- Environment --------------------
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    # -------------------- App Server --------------------
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8081, env="PORT")

    # -------------------- Output --------------------
    output_dir: str = Field(default="./data/output", env="OUTPUT_DIR")
    default_format: str = Field(default="csv", env="DEFAULT_FORMAT")

    # -------------------- Limits --------------------
    max_rows_per_request: int = Field(default=5000, env="MAX_ROWS_PER_REQUEST")

    # -------------------- Logging --------------------
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")

    # -------------------- AWS Core --------------------
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_session_token: Optional[str] = Field(default=None, env="AWS_SESSION_TOKEN")

    # -------------------- Bedrock --------------------
    bedrock_model: str = Field(
        default="amazon.nova-pro-v1:0",
        env="BEDROCK_MODEL"
    )

    bedrock_embedding_model: Optional[str] = Field(
        default=None,
        env="BEDROCK_EMBEDDING_MODEL"
    )

    # -------------------- Validators --------------------

    @field_validator("environment")
    def validate_environment(cls, v):
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    @field_validator("log_level")
    def validate_log_level(cls, v):
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v

    @field_validator("default_format")
    def validate_format(cls, v):
        allowed = ["csv", "parquet", "json"]
        if v not in allowed:
            raise ValueError(f"default_format must be one of {allowed}")
        return v

    # -------------------- AWS Credential Setup --------------------

    def configure_aws_credentials(self):
        """
        Configure AWS credentials if provided via environment.
        If not provided, boto3 will fallback to:
        - AWS CLI config
        - IAM role
        """

        if self.aws_access_key_id and self.aws_secret_access_key:
            os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key

            if self.aws_session_token:
                os.environ["AWS_SESSION_TOKEN"] = self.aws_session_token

            os.environ["AWS_DEFAULT_REGION"] = self.aws_region

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance"""
    return Settings()


# Optional backward compatibility
settings = get_settings()