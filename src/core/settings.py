from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    kaggle_username: str = Field(..., description="Kaggle username")
    kaggle_api_key: str = Field(..., description="Kaggle API key")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "forbid"
