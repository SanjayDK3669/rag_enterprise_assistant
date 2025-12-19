from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.models.datatypes import DocumentChunk
from app.config.constants import CHUNK_OVERLAP, CHUNK_SIZE

class TextChunkSplitter:
    """
    Splits documents into smaller overlapping chunks 
    suitable for embedding and retrieval.
    """
    
    def __init__(self,
                chunk_size: int = CHUNK_SIZE,
                chunk_overlap: int = CHUNK_OVERLAP):
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size= chunk_size,
            chunk_overlap= chunk_overlap)
    
    def split(self, documents: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Split document pages into chunks.

        Args:
            document (List[DocumentChunk]): Raw Documents

        Returns:
            List[DocumentChunk]: Chunked documents.
        """
        split_chunks : List[DocumentChunk] = []
        
        for doc in documents:
            texts = self.splitter.split_text(doc.text)
            
            for text in texts:
                split_chunks.append(
                    DocumentChunk(
                        text= text,
                        metadata= doc.metadata
                    )
                )
                
        return split_chunks
    