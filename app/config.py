from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    pinecone_api_key: str = "e01f7779-c3a4-459b-96ba-aef09069c5c4"
    pinecone_environment: str = "us-east-1"
    model_name: str = "all-MiniLM-L6-v2"
    index_name: str = "resume-index"
    llm_model_name: str = "mixtral-8x7b-32768"  # Groq model name

settings = Settings()
