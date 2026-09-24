from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Splits documents into smaller text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)
