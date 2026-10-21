"""Process configuration for the multi-agent platform."""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from research_platform.platform import (
    DEFAULT_INPUT_USD_PER_TOKEN,
    DEFAULT_OUTPUT_USD_PER_TOKEN,
    load_vault_key,
)

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PLATFORM_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "dev"
    service_name: str = "multi-agent-research-platform"
    llm_mode: str = "auto"
    llm_api_key: SecretStr | None = None
    vault_secret_path: str | None = None
    max_cost_usd: float = 2.0
    timeout_s: float = 180.0
    agent_timeout_s: float = 20.0
    specialist_delay_s: float = 0.0
    fail_agent: str = ""
    input_usd_per_token: float = DEFAULT_INPUT_USD_PER_TOKEN
    output_usd_per_token: float = DEFAULT_OUTPUT_USD_PER_TOKEN
    llm_timeout_s: float = 2.0
    rate_limit_per_minute: int = 30

    @property
    def resolved_api_key(self) -> SecretStr | None:
        env_key = self.llm_api_key.get_secret_value() if self.llm_api_key else None
        return load_vault_key(self.vault_secret_path, app_env=self.app_env, env_key=env_key)
