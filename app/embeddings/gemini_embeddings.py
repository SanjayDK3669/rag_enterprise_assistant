from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.models.datatypes import EmbeddedChunk, DocumentChunk
from app.config.settings import GeminiConfig

class GeminiEmbeddingService:
    """
    Generate embeddings using Gemini embedding model.
    """
    def __init__(self):
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model= GeminiConfig.embedding_model,
            google_api_key= GeminiConfig.api_key
        )
    
    def embed_document(self, 
                       chunks: List[DocumentChunk]) -> List[EmbeddedChunk]:
        """
        Convert document chunks into vector embedding.
        Args:
            chunks (List[DocumentChunk]): 

        Returns:
            List[EmbeddedChunk]: _description_
        """
        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding_model.embed_documents(texts= texts)
        
        embedded_chunks : List[EmbeddedChunk] = []
        
        for chunk, vector in zip(chunks, vectors):
            embedded_chunks.append(
                EmbeddedChunk(
                    text= chunk.text,
                    embedding=vector,
                    metadata=chunk.metadata
                )
            )
            
        return embedded_chunks
        