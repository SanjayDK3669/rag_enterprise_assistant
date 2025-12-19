from app.loaders.pdf_loader import PDFDocumentLoader
from app.splitters.text_splitter import TextChunkSplitter
from app.embeddings.gemini_embeddings import GeminiEmbeddingService
from app.vectorstore.chroma_store import ChromaVectorStore

loader = PDFDocumentLoader(pdf_directory= "data/raw_pdfs")
docs = loader.load()

splitter = TextChunkSplitter()
chunks = splitter.split(documents= docs)

embedding_service = GeminiEmbeddingService()
embedded = embedding_service.embed_document(chunks=chunks)

store = ChromaVectorStore()
store.create_or_load()
store.add_embeddings(embedded)


print("Embedding stored in Chroma")