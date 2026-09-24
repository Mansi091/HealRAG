from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .prompt import get_prompt_template, format_docs

def build_rag_chain(retriever, model_name: str = "openai/gpt-4o-mini", api_key: str = None, base_url: str = None):
    """Builds the LCEL RAG chain."""
    llm_kwargs = {"temperature": 0.0, "api_key": api_key}
    if base_url:
        llm_kwargs["model"] = model_name
        llm_kwargs["openai_api_base"] = base_url
    else:
        llm_kwargs["model"] = "gpt-4o-mini"

    llm = ChatOpenAI(**llm_kwargs)
    prompt = get_prompt_template()

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
