# Project Management & Software Architecture Guidelines

## Core Principles
1. Maintain Clean Code & Clear Separation of Concerns: Separate document ingestion, vector storage, retrieval, generation, evaluation, and API handlers.
2. Bounded Retries: Never create infinite loops in automated healing systems. Limit retries (e.g. MAX_RETRIES = 2) to ensure system reliability and bound execution cost.
3. Grounding Verification: Always verify that LLM outputs strictly derive from the supplied context before serving final answers to users.
4. Continuous Quality Monitoring: Maintain a golden evaluation dataset to track precision, recall, faithfulness, and answer relevancy.

## Health and Recovery
When quality degrades below configured thresholds (e.g., RAGAS score < 0.70), the system must trigger automated recovery workflows such as index re-building or fallback retrieval routines, followed by re-evaluation to verify recovery.
