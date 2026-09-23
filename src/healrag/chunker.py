from typing import List, Dict, Any, Optional
import uuid


class Document:

    def __init__(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.doc_id = doc_id if doc_id is not None else str(uuid.uuid4())
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self) -> str:
        return f"Document(doc_id={self.doc_id!r}, text={self.text[:30]!r}...)"


class Chunk:

    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any],
        start_char: int,
        end_char: int,
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.text = text
        self.metadata = metadata
        self.start_char = start_char
        self.end_char = end_char

    def __repr__(self) -> str:
        return f"Chunk(chunk_id={self.chunk_id!r}, text={self.text[:30]!r}...)"


class TextChunker:

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, document: Document) -> List[Chunk]:
        text = document.text.strip()
        if not text:
            return []

        chunks: List[Chunk] = []
        step = self.chunk_size - self.chunk_overlap

        for index, start in enumerate(range(0, len(text), step)):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}_chunk_{index}",
                        doc_id=document.doc_id,
                        text=chunk_text,
                        metadata={**document.metadata, "chunk_index": index},
                        start_char=start,
                        end_char=end,
                    )
                )

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        return all_chunks
