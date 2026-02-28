from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME : str
    API_STR : str
    VERSION : str
    
    POSTGRES_USER : str
    POSTGRES_PASSWORD : str
    POSTGRES_DB : str
    POSTGRES_HOST : str
    POSTGRES_PORT : int
    
    DATABASE_URL : str
    
    JWT_SECRET_KEY : str
    JWT_ALGORITHM : str
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES : int
    
    LLM_MODEL : str
    EMBEDDING_MODEL : str
    RERANKER_MODEL : str
    GEMINI_API_KEY : str
    OLLAMA_BASE_URL : str
    EVAL_LLM_MODEL : str
    
    QDRANT_URL : str
    QDRANT_COLLECTION_NAME : str
    
    CHUNK_SIZE : int
    CHUNK_OVERLAP : int
    RETRIEVAL_TOP_K : int
    
    MLFLOW_TRACKING_URI : str
    MLFLOW_EXPERIMENT_NAME : str
    
    GRAFANA_ADMIN_PASSWORD : str
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        case_sensitive=True,
        extra="ignore"
    )
    
    
settings = Settings()