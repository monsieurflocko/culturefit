from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Hier explizit deine lokale Dev-URL eintragen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Request-Modell richtig definiert
class MatchRequest(BaseModel):
    user_answers: List[str]
    openai_key: str

# ✅ Matching-Funktion
def compute_matching(user_answers, company_data_path, min_results, score_threshold, openai_key):
    client = OpenAI(api_key=openai_key)
    embedding_text = " ".join(user_answers)

    response = client.embeddings.create(
        input=embedding_text,
        model="text-embedding-ada-002"
    )
    user_vector = np.array(response.data[0].embedding)

    with open(company_data_path, "r", encoding="utf-8") as f:
        company_data = json.load(f)

    company_vectors = np.array([entry["vector"] for entry in company_data])
    similarities = cosine_similarity([user_vector], company_vectors)[0]

    matches = sorted(
        [
            {
                "id": entry["id"],
                "name": entry["name"],
                "description_short": entry.get("description_short", ""),
                "score": float(sim)
            }
            for entry, sim in zip(company_data, similarities)
        ],
        key=lambda x: x["score"],
        reverse=True
    )

    filtered = [m for m in matches if m["score"] >= score_threshold]
    if len(filtered) < min_results:
        filtered = matches[:min_results]

    return filtered

# ✅ Hier ist der Schlüssel: Request-Modell korrekt referenzieren
@app.post("/match")
async def match_companies(request: MatchRequest):
    results = compute_matching(
        user_answers=request.user_answers,
        company_data_path="embedding_vectors.json",
        min_results=3,
        score_threshold=0.75,
        openai_key=request.openai_key
    )
    return {"matches": results}
