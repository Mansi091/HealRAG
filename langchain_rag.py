import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None

if not api_key:
    print("⚠️ Warning: Neither OPENROUTER_API_KEY nor OPENAI_API_KEY found in environment or .env file.")

# --------------------------------------------------
# 2. Load document
# --------------------------------------------------
PDF_PATH = "data/documents/document.pdf"

if os.path.exists(PDF_PATH):
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF")
else:
    print(f"PDF not found at {PDF_PATH}. Creating sample memory documents...")
    from langchain_core.documents import Document
    documents = [
        Document(page_content="Adults should aim for at least 150 minutes of moderate aerobic activity weekly for cardiovascular health.", metadata={"page": 0, "source": "sample"}),
        Document(page_content="Type 2 diabetes can often be prevented by increasing soluble dietary fiber intake and minimizing refined sugar.", metadata={"page": 1, "source": "sample"}),
        Document(page_content="Adequate daily water intake (2.5 to 3.5 liters) is essential for kidney health and preventing kidney stones.", metadata={"page": 2, "source": "sample"})
    ]

# --------------------------------------------------
# 3. Split document into chunks
# --------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# --------------------------------------------------
# 4. Create embeddings
# --------------------------------------------------
embedding_kwargs = {"model": "text-embedding-3-small"}
if api_key:
    embedding_kwargs["api_key"] = api_key
if base_url:
    embedding_kwargs["openai_api_base"] = base_url

embeddings = OpenAIEmbeddings(**embedding_kwargs)

# --------------------------------------------------
# 5. Store embeddings in ChromaDB
# --------------------------------------------------
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="healrag_documents",
    persist_directory="./chroma_db"
)

print("Documents stored in ChromaDB")

# --------------------------------------------------
# 6. Create retriever
# --------------------------------------------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# --------------------------------------------------
# 7. Create LLM
# --------------------------------------------------
llm_kwargs = {
    "model": "openai/gpt-4o-mini" if base_url else "gpt-4o-mini",
    "temperature": 0
}
if api_key:
    llm_kwargs["api_key"] = api_key
if base_url:
    llm_kwargs["openai_api_base"] = base_url

llm = ChatOpenAI(**llm_kwargs)

# --------------------------------------------------
# 8. Create RAG prompt
# --------------------------------------------------
prompt = ChatPromptTemplate.from_template("""
You are a helpful document question-answering assistant.

Answer the question using ONLY the provided context.

If the answer is not present in the context,
say "I don't have enough information in the document."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
""")

# --------------------------------------------------
# 9. Ask a question
# --------------------------------------------------
query_text = "How many minutes of exercise are recommended for heart health?"
print(f"\nQuestion: {query_text}")

retrieved_docs = retriever.invoke(query_text)
print(f"\nRetrieved {len(retrieved_docs)} chunks")

# --------------------------------------------------
# 10. Build context
# --------------------------------------------------
context = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)

# --------------------------------------------------
# 11. Send context + question to LLM
# --------------------------------------------------
messages = prompt.invoke({
    "context": context,
    "question": query_text
})

response = llm.invoke(messages)

# --------------------------------------------------
# 12. Display answer
# --------------------------------------------------
print("\n================ ANSWER ================\n")
print(response.content)

# --------------------------------------------------
# 13. Display sources
# --------------------------------------------------
print("\n================ SOURCES ================\n")
for doc in retrieved_docs:
    print(
        f"Page: {doc.metadata.get('page', 0) + 1} | Source: {doc.metadata.get('source', 'Unknown')}"
    )
