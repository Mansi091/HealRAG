def get_retriever(vectorstore, top_k: int = 4):
    """Creates a similarity retriever from a vectorstore."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
