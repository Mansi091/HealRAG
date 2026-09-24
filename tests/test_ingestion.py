import pytest
from langchain_core.documents import Document
from core.ingestion.loader import DocumentLoader
from core.ingestion.splitter import DocumentSplitter


def test_document_loader(tmp_path):
    # Create temp sample file
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.txt"
    p.write_text("HealRAG is a self-healing RAG system.")

    loader = DocumentLoader(docs_dir=str(d))
    docs = loader.load_documents()

    assert len(docs) == 1
    assert "HealRAG" in docs[0].page_content
    assert docs[0].metadata.get("filename") == "test.txt"


def test_document_splitter():
    docs = [Document(page_content="Word " * 500, metadata={"source": "test.txt"})]
    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    assert len(chunks) > 1
    assert chunks[0].metadata.get("source") == "test.txt"
