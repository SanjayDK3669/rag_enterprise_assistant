from typing import List
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from app.config.settings import ChromaConfig
from app.models.datatypes import EmbeddedChunk

class ChromaVectorStore:
    """
    Models storage and retrieval of embedding using Chroma.
    """
    def __init__(self):
        self.persist_directory = ChromaConfig.persist_directory
        self.collection_name = ChromaConfig.collection_name
        self.vectorstore = None
    
    def create_or_load(self) -> None:
        """
        Create or load existing chroma collection.
        """
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory= self.persist_directory
        )

    def add_embeddings(self, chunks: List[EmbeddedChunk]) -> None:
        """Store embedded chunks into chromaDB.

        Args:
            chunks (List[EmbeddedChunk]): _description_
        """
        documents = [
            Document(
                page_content = chunk.text,
                metadata = chunk.metadata
            ) 
            for chunk in chunks
        ]
        
        embeddings = [chunk.embedding for chunk in chunks]
        
        self.vectorstore.add_embeddings(
            texts = [doc.page_content for doc in documents],
            metadatas = [doc.metadata for doc in documents],
            embeddings = embeddings
        )
        
        self.vectorstore.persist()
    
    def similarity_search(self,
                          query_embedding: List[float],
                          top_k: int ):
        """
        perform similarity search.
        Args:
            query_embedding (List[float]): _description_
            top_k (int): _description_
        """
        return self.vectorstore.similarity_search_by_vector(
            embedding=query_embedding,
            k = top_k
        )
        
