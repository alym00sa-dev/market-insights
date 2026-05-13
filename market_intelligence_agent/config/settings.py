from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    LLM_PROVIDER: str = "claude"
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-sonnet-4-6"
    OPENAI_MODEL: str = "gpt-4o"

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"

    # News APIs
    NEWS_API_KEY: str = ""
    GUARDIAN_API_KEY: str = ""
    NYT_API_KEY: str = ""
    ARTIFICIAL_ANALYSIS_API_KEY: str = ""

    # Scraping
    SCRAPE_INTERVAL_HOURS: int = 4
    MAX_ARTICLES_PER_SOURCE: int = 50

    # App
    LOG_LEVEL: str = "INFO"

    model_config = {
        "env_file": Path(__file__).parent.parent.parent / ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
