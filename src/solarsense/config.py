from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SOLARSENSE_", env_file=str(ROOT / ".env"), extra="ignore"
    )

    # Plant (simulated by default; swap the loader for real SCADA data)
    plant_name: str = "Pune Solar Park (simulated)"
    capacity_kw: float = 5000.0
    latitude: float = 18.52
    longitude: float = 73.86
    sim_start: str = "2026-01-01"
    sim_days: int = 150
    train_days: int = 90
    seed: int = 42

    # Business assumptions (illustrative, configurable)
    tariff_inr_per_kwh: float = 3.0
    dsm_tolerance_pct: float = 10.0
    dsm_penalty_multiplier: float = 1.0
    cleaning_cost_inr: float = 20000.0

    # Ops
    artifact_dir: Path = ROOT / "artifacts"
    admin_api_key: str = ""  # empty = /api/retrain disabled
    groq_api_key: str = Field(
        "", validation_alias=AliasChoices("GROQ_API_KEY", "SOLARSENSE_GROQ_API_KEY")
    )
    groq_model: str = Field(
        "openai/gpt-oss-120b", validation_alias=AliasChoices("GROQ_MODEL", "SOLARSENSE_GROQ_MODEL")
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
