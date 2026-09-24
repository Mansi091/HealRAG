import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_core.documents import Document

def load_documents(docs_dir: str = "data/documents"):
    """Loads PDF and TXT documents from docs_dir or returns fallback sample documents."""
    documents = []

    if os.path.exists(docs_dir):
        pdf_loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        txt_loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents.extend(pdf_loader.load())
        documents.extend(txt_loader.load())

    if not documents:
        documents = [
            Document(
                page_content="Artificial Intelligence (AI) refers to computer systems designed to perform tasks requiring human intelligence.",
                metadata={"source": "ai_overview.txt", "page": 0}
            ),
            Document(
                page_content="Project Management Guidelines: Successful project execution requires clear goals, stakeholder communication, and risk management.",
                metadata={"source": "project_guidelines.pdf", "page": 1}
            )
        ]

    return documents
