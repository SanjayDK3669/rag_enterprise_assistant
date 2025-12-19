from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DocumentChunk:
    """
    Represents a chunk of a document after splitting.
    Used before embedding and vector storage.
    """
    text: str
    metadata: Dict[str, Any]
    
@dataclass
class EmbeddedChunk:
    """
    Represent a chunk with its vector embedding.
    """
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    
@dataclass
class UserQuery:
    """
    Input query from the user.
    """
    question: str
    top_k: int = 4
    
@dataclass
class RetrieverDocument:
    """
    Document returned from vector search.
    """
    content: str
    metadata: Dict[str, Any]
    score: float
    