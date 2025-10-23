from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # directories
    data_dir: Path = Field(default=Path("./data"))
    bronze_dir: Path = Field(default=Path("./data/bronze"))
    silver_dir: Path = Field(default=Path("./data/silver"))
    gold_dir: Path = Field(default=Path("./data/gold"))

    # inputs/outputs
    csv_path: Path = Field(default=Path("./data/raw/PS_20174392719_1491204439457_log.csv"))
    sqlite_url: str = Field(default="sqlite:///./reports/daily_summary.sqlite")
    model_path: Path = Field(default=Path("./artifacts/model.lgb"))

    # env threshold optional
    alert_threshold: float | None = None

    # pydantic-settings v2 way to point at .env
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
