import os
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, DirectoryLoader
from langchain_core.documents import Document
from config import settings

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads PDF, TXT, and Markdown documents while preserving metadata."""

    def __init__(self, docs_dir: str = settings.DOCS_DIR):
        self.docs_dir = docs_dir

    def load_documents(self) -> List[Document]:
        """Scans docs_dir and loads supported document formats."""
        documents: List[Document] = []

        if not os.path.exists(self.docs_dir):
            logger.warning(f"Directory {self.docs_dir} does not exist. Creating...")
            os.makedirs(self.docs_dir, exist_ok=True)
            return documents

        # Load PDFs
        pdf_loader = DirectoryLoader(
            self.docs_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=False
        )
        try:
            pdf_docs = pdf_loader.load()
            for doc in pdf_docs:
                filename = os.path.basename(doc.metadata.get("source", "unknown.pdf"))
                doc.metadata["filename"] = filename
            documents.extend(pdf_docs)
            logger.info(f"Loaded {len(pdf_docs)} PDF page documents.")
        except Exception as e:
            logger.warning(f"Error loading PDFs: {e}")

        # Load TXT files
        txt_loader = DirectoryLoader(
            self.docs_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=False
        )
        try:
            txt_docs = txt_loader.load()
            for doc in txt_docs:
                filename = os.path.basename(doc.metadata.get("source", "unknown.txt"))
                doc.metadata["filename"] = filename
                doc.metadata["page"] = 0
            documents.extend(txt_docs)
            logger.info(f"Loaded {len(txt_docs)} TXT documents.")
        except Exception as e:
            logger.warning(f"Error loading TXT files: {e}")

        # Load Markdown files
        md_loader = DirectoryLoader(
            self.docs_dir,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=False
        )
        try:
            md_docs = md_loader.load()
            for doc in md_docs:
                filename = os.path.basename(doc.metadata.get("source", "unknown.md"))
                doc.metadata["filename"] = filename
                doc.metadata["page"] = 0
            documents.extend(md_docs)
            logger.info(f"Loaded {len(md_docs)} Markdown documents.")
        except Exception as e:
            logger.warning(f"Error loading Markdown files: {e}")

        logger.info(f"Total loaded documents: {len(documents)}")
        return documents
