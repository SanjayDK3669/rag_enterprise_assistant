from typing import List
from langchain_community.document_loaders import PyPDFLoader
from app.models.datatypes import DocumentChunk
import os

class PDFDocumentLoader:
    """
    Load PDF Documents from a directory and convert them into DocumentChunk objects.
    """
    def __init__(self, pdf_directory: str = None): 
        self.pdf_directory = pdf_directory
        
    def load(self)-> List[DocumentChunk]:
        """
        Load all PDFS from the directory

        Returns:
            List[DocumentChunk]: _description_
        """
        chunks: List[DocumentChunk] = []
        for filename in os.listdir(self.pdf_directory):
            if not filename.lower().endswith(".pdf"):
                continue
            
            file_path = os.path.join(self.pdf_directory, filename)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            for doc in documents:
                chunks.append(
                    DocumentChunk(
                        text= doc.page_content,
                        metadata={
                            "source": filename,
                            "page": doc.metadata.get("page", None)
                        }
                    )
                )
        return chunks