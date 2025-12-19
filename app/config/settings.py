from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class GeminiConfig:
    api_key: str = os.getenv("GEMINI_API_KEY")
    chat_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/embedding-001"

@dataclass(frozen=True)
class ChromaConfig:
    persist_directory: str = "data/chroma_db"
    collection_name: str = "enterprise_docs"