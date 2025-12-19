from app.loaders.pdf_loader import PDFDocumentLoader
from app.splitters.text_splitter import TextChunkSplitter

loader = PDFDocumentLoader(pdf_directory= "data/raw_pdfs")
docs = loader.load()

splitter = TextChunkSplitter()
chunks = splitter.split(documents= docs)

print(f"Chunks created : {len(chunks)}")
print(chunks[0])