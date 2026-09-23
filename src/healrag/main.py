"""
Main demonstration entrypoint for the Normal RAG Pipeline with OpenRouter integration.
"""
import os
import sys

# Load .env file if available
env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

from healrag import RAGPipeline, Document, SimpleTFIDFEmbedder, OpenAIEmbedder


# Sample dataset: Medical & General Health Guidelines
SAMPLE_DOCUMENTS = [
    Document(
        text="""Cardiovascular Health & Exercise Guidelines:
Regular physical activity is vital for maintaining cardiovascular health. Adults should aim for at least 150 minutes of moderate-intensity aerobic physical activity or 75 minutes of vigorous-intensity activity weekly. Aerobic exercises such as brisk walking, swimming, cycling, and running strengthen the heart muscle, lower blood pressure, and reduce LDL cholesterol levels.""",
        metadata={"source": "cardio_guidelines_2026.pdf", "category": "Cardiology"}
    ),
    Document(
        text="""Type 2 Diabetes Prevention & Nutrition:
Type 2 Diabetes can often be managed or prevented through dietary modifications and weight management. Key nutritional strategies include increasing soluble dietary fiber intake (whole grains, legumes, vegetables), minimizing refined carbohydrates and sugar-sweetened beverages, and maintaining a balanced glycemic index in daily meals. Regular monitoring of HbA1c levels is recommended.""",
        metadata={"source": "diabetes_nutrition_manual.pdf", "category": "Endocrinology"}
    ),
    Document(
        text="""Hydration and Kidney Function:
Adequate daily water intake is essential for proper renal filtration and metabolic waste excretion. Healthy adults should generally consume between 2.5 to 3.5 liters of fluid per day depending on physical climate and activity level. Dehydration increases the risk of nephrolithiasis (kidney stones) and acute kidney injury.""",
        metadata={"source": "nephrology_basics.pdf", "category": "Nephrology"}
    )
]


def main():
    # Configure stdout to handle UTF-8 cleanly
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    print("=" * 60)
    print("         NORMAL RAG PIPELINE DEMONSTRATION")
    print("=" * 60)

    # Check for OpenRouter / OpenAI API key in environment
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if api_key:
        print("\n[KEY] API Key loaded successfully!")
        print("      Embedder: SimpleTFIDFEmbedder (Fast local vector retrieval)")
        print("      LLM: OpenRouter (openai/gpt-4o-mini)")
        pipeline = RAGPipeline(
            embedder=SimpleTFIDFEmbedder(),
            api_key=api_key,
            model="openai/gpt-4o-mini",
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        print("\n[*] Running in Offline / Open-Source Mode (No API key found).")
        print("    Using local TF-IDF vector embeddings for similarity search.")
        pipeline = RAGPipeline(embedder=SimpleTFIDFEmbedder())

    # Step 1: Ingest Documents
    print("\n[Step 1: Document Ingestion & Chunking]")
    num_chunks = pipeline.ingest_documents(SAMPLE_DOCUMENTS)
    print(f"[+] Successfully indexed {len(SAMPLE_DOCUMENTS)} documents into {num_chunks} text chunks.")

    # Step 2: Querying
    sample_queries = [
        "How many minutes of exercise are recommended for heart health?",
        "What dietary changes help prevent Type 2 Diabetes?",
        "What are the risks of dehydration for kidney health?"
    ]

    for i, query in enumerate(sample_queries, 1):
        print("\n" + "-" * 60)
        print(f"QUERY {i}: {query}")
        print("-" * 60)

        result = pipeline.query(query, top_k=2)

        print("\n[+] RETRIEVED TOP-K CONTEXT CHUNKS:")
        for idx, (chunk, score) in enumerate(result["retrieved_chunks"], 1):
            src = chunk.metadata.get("source", "Unknown")
            print(f"  Chunk #{idx} [Source: {src} | Similarity Score: {score:.4f}]")
            print(f"  \"{chunk.text[:140]}...\"\n")

        print("[+] GENERATED RAG RESPONSE (OpenRouter GPT-4o-mini):")
        print(result["answer"])

    print("\n" + "=" * 60)
    print("Pipeline Execution Completed Successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
