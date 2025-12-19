from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RAGResponse:
    """
    Final response returned by the RAG pipeline
    """
    answer: str
    sources: List[str]
    confidence: Optional[float] = None 