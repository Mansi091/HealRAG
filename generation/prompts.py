from langchain_core.prompts import ChatPromptTemplate

# 1. Answer Generation Prompt
ANSWER_GENERATION_PROMPT = ChatPromptTemplate.from_template("""
You are a reliable AI assistant. Answer the user question based strictly on the provided context below.

Rules:
1. Use ONLY the provided context to answer the question.
2. Do NOT invent, assume, or extrapolate information.
3. If the context does not contain sufficient information to answer the question, clearly state: "I do not have enough information in the provided context to answer this question."
4. Maintain factual precision and preserve source citations.

Context:
{context}

Question:
{question}

Answer:""")


# 2. Document Relevance Grading Prompt
RELEVANCE_GRADER_PROMPT = ChatPromptTemplate.from_template("""
You are a quality grader evaluating whether a retrieved document is relevant to a user question.

Question: {question}

Document Content:
{document}

Instructions:
Evaluate if the document contains keywords, concepts, or semantic information related to the question.
Respond with a JSON object containing:
- "relevant": true if the document is relevant, false otherwise.
- "score": float between 0.0 and 1.0 indicating relevance confidence.
- "reason": brief explanation of your rating.
""")


# 3. Query Rewriting Prompt
QUERY_REWRITER_PROMPT = ChatPromptTemplate.from_template("""
You are an expert query optimizer for vector search systems.

Original User Query: {question}

Context / Issues Found:
Previous retrieval failed to yield relevant context. Rephrase the query to improve semantic retrieval quality. Make it clearer, more specific, and keyword-rich without changing the original intent.

Respond with a JSON object containing:
- "rewritten_query": the optimized query string.
- "reason": brief explanation of changes made.
""")


# 4. Grounding / Hallucination Checking Prompt
GROUNDING_CHECKER_PROMPT = ChatPromptTemplate.from_template("""
You are an AI fact-checking evaluator inspecting whether a generated answer is strictly supported by the provided context.

Context:
{context}

Question:
{question}

Generated Answer:
{answer}

Instructions:
Evaluate if every factual statement in the generated answer is directly supported by the context.
Respond with a JSON object containing:
- "grounded": true if fully supported by context, false if hallucinated or ungrounded.
- "score": float between 0.0 and 1.0 indicating grounding confidence.
- "reason": detailed explanation of your judgment.
""")


# 5. Recovery Diagnosis Prompt
RECOVERY_DIAGNOSIS_PROMPT = ChatPromptTemplate.from_template("""
You are an automated system health diagnostic agent.

Current System Metrics:
{metrics}

Baseline Metrics:
{baseline}

Degraded Components / Logs:
{logs}

Instructions:
Diagnose the root cause of degradation and select the recommended recovery action from:
["rebuild_index", "reindex_documents", "reset_vectorstore", "fallback_retrieval"]

Respond with a JSON object containing:
- "diagnosis": brief diagnostic summary.
- "recommended_action": choice from recommended actions list.
- "confidence": float between 0.0 and 1.0.
""")
