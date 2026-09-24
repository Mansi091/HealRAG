import os
import sys
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProductionRAG:
    """
    Production-Grade RAG Pipeline using LangChain, ChromaDB, and OpenRouter / OpenAI.
    Features LCEL chains, persistent vector storage, multi-format loaders, and metadata sources.
    """

    def __init__(
        self,
        docs_dir: str = "data/documents",
        chroma_db_dir: str = "./chroma_db",
        collection_name: str = "rag_production",
        model_name: str = "openai/gpt-4o-mini",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 4
    ):
        load_dotenv()
        self.docs_dir = docs_dir
        self.chroma_db_dir = chroma_db_dir
        self.collection_name = collection_name
        self.top_k = top_k

        self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key missing! Please set OPENROUTER_API_KEY or OPENAI_API_KEY in your .env file.")

        self.base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )


        llm_kwargs = {
            "model": model_name if self.base_url else "gpt-4o-mini",
            "temperature": 0.0,
            "api_key": self.api_key
        }
        if self.base_url:
            llm_kwargs["openai_api_base"] = self.base_url
        self.llm = ChatOpenAI(**llm_kwargs)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.vectorstore: Optional[Chroma] = None
        self.retriever = None
        self.chain = None

    def load_documents(self) -> List[Any]:
        """Load documents from directory (PDFs, TXT) or generate sample fallback documents."""
        documents = []

        if os.path.exists(self.docs_dir):
            logger.info(f"Loading documents from directory: {self.docs_dir}")
            pdf_loader = DirectoryLoader(self.docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
            documents.extend(pdf_loader.load())

            txt_loader = DirectoryLoader(self.docs_dir, glob="**/*.txt", loader_cls=TextLoader)
            documents.extend(txt_loader.load())

        if not documents:
            logger.info(f"No documents found in '{self.docs_dir}'. Using general sample dataset.")
            from langchain_core.documents import Document
            documents = [
                Document(
                    page_content="Artificial Intelligence (AI) refers to computer systems designed to perform tasks that typically require human intelligence, such as language processing, data analysis, and decision-making.",
                    metadata={"source": "ai_overview.txt", "page": 0}
                ),
                Document(
                    page_content="Project Management Guidelines: Successful project execution requires defining clear goals, maintaining regular stakeholder communication, managing risks, and monitoring key milestones.",
                    metadata={"source": "project_guidelines.pdf", "page": 1}
                ),
                Document(
                    page_content="Information Security Best Practices: Organizations should enforce strong access controls, encryption, regular software updates, and data backups to prevent unauthorized access.",
                    metadata={"source": "security_policy.pdf", "page": 2}
                )
            ]

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def initialize_pipeline(self):
        """Index documents into ChromaDB and build the LCEL RAG Chain."""
        logger.info("Indexing documents into ChromaDB vector store...")
        documents = self.load_documents()
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from documents.")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.chroma_db_dir
        )
        logger.info("ChromaDB vector store initialized successfully.")

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )

        prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the user question based strictly on the provided context below.

Rules:
1. Use ONLY the provided context to answer the question.
2. If the answer is not present in the context, respond with "I do not have enough information in the provided context to answer this question."
3. Keep your response concise, factual, and accurate.

Context:
{context}

Question:
{question}

Answer:""")

        def format_docs(docs):
            return "\n\n".join(
                f"[Source: {d.metadata.get('source', 'Unknown')} (Page {d.metadata.get('page', 0) + 1})]\n{d.page_content}"
                for d in docs
            )

        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info("LCEL RAG Pipeline Chain ready.")

    def query(self, question: str) -> Dict[str, Any]:
        """Execute query against RAG pipeline and return answer with source attribution."""
        if not self.chain or not self.retriever:
            raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")

        logger.info(f"Executing Query: '{question}'")
        retrieved_docs = self.retriever.invoke(question)
        answer = self.chain.invoke(question)

        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0) + 1,
                "snippet": doc.page_content[:120] + "..."
            }
            for doc in retrieved_docs
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }


def main():
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    print("=" * 60)
    print("    PRODUCTION LANGCHAIN RAG PIPELINE (ChromaDB + LCEL)")
    print("=" * 60)

    rag = ProductionRAG()
    rag.initialize_pipeline()

    sample_questions = [
        "What is artificial intelligence?",
        "What are the guidelines for successful project management?",
        "What are the best practices for information security?"
    ]

    for q in sample_questions:
        print("\n" + "-" * 60)
        result = rag.query(q)
        print(f"\n[QUESTION] {result['question']}")
        print(f"\n[ANSWER]\n{result['answer']}")
        print("\n[SOURCES]")
        for s in result["sources"]:
            print(f"  * Source: {s['source']} (Page {s['page']})")

    print("\n" + "=" * 60)
    print("Pipeline Execution Completed Successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
