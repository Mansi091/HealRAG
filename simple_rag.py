import os
import numpy as np
from openai import OpenAI

# 1. Load API Key from environment or .env file
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key and os.path.exists(".env"):
    with open(".env", encoding="utf-8") as f:
        for line in f:
            if line.startswith("OPENROUTER_API_KEY="):
                api_key = line.split("=", 1)[1].strip()

# 2. Documents (Your Knowledge Base)
documents = [
    "Adults should aim for at least 150 minutes of moderate aerobic activity weekly for cardiovascular health.",
    "Type 2 diabetes can be prevented by increasing soluble dietary fiber and minimizing refined sugar.",
    "Drinking 2.5 to 3.5 liters of water daily helps prevent kidney stones and supports renal health."
]

# 3. Simple Embedder (Converts text into numbers/vectors)
def text_to_vector(text: str, vocab_size: int = 256) -> np.ndarray:
    vec = np.zeros(vocab_size, dtype=np.float32)
    for word in text.lower().split():
        vec[hash(word) % vocab_size] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# Convert all documents to vectors
doc_vectors = np.array([text_to_vector(doc) for doc in documents])

# 4. User Query
question = "How many minutes of exercise are recommended for heart health?"

# 5. Retrieve Most Relevant Document (Cosine Similarity Search)
query_vector = text_to_vector(question)
similarities = np.dot(doc_vectors, query_vector)
best_match_idx = np.argmax(similarities)
retrieved_context = documents[best_match_idx]

print(f"Question: {question}")
print(f"Retrieved Context: {retrieved_context}\n")

# 6. Generate Answer using OpenRouter LLM
if api_key:
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    prompt = f"Answer using ONLY this context:\n{retrieved_context}\n\nQuestion: {question}"
    
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print("AI Answer:", response.choices[0].message.content)
else:
    print("No API Key found. Add OPENROUTER_API_KEY to your .env file.")
