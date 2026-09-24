from langchain_core.prompts import ChatPromptTemplate

def get_prompt_template():
    """Returns ChatPromptTemplate for RAG."""
    return ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the user question based strictly on the provided context below.

Rules:
1. Use ONLY the provided context to answer the question.
2. If the answer is not present in the context, respond with "I do not have enough information in the provided context to answer this question."

Context:
{context}

Question:
{question}

Answer:""")

def format_docs(docs):
    """Formats retrieved document snippets for the prompt context."""
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'Unknown')} (Page {d.metadata.get('page', 0) + 1})]\n{d.page_content}"
        for d in docs
    )
