import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

# Load environment variables
load_dotenv()

@dataclass
class DatabricksConfig:
    host: str = os.getenv("DATABRICKS_HOST", "")
    token: str = os.getenv("DATABRICKS_TOKEN", "")
    foundation_model: str = os.getenv("FOUNDATION_MODEL_ENDPOINT", "databricks-llama-4-maverick")
    embedding_model: str = os.getenv("EMBEDDING_MODEL_ENDPOINT", "databricks-bge-large-en")
    fallback_embedding: str = os.getenv("FALLBACK_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

@dataclass
class PerformanceConfig:
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", "1500"))
    cache_size_limit: int = int(os.getenv("CACHE_SIZE_LIMIT", "100"))
    response_cache_ttl: int = int(os.getenv("RESPONSE_CACHE_TTL", "1800"))

@dataclass
class AppConfig:
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
# Global config instances
databricks_config = DatabricksConfig()
performance_config = PerformanceConfig()
app_config = AppConfig()