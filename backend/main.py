# main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS erlauben (Frontend darf Anfragen schicken)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # später evtl. auf deine Vercel-URL beschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/match")
async def match(request: Request):
    data = await request.json()
    return {
        "message": "Anfrage empfangen!",
        "deine_daten": data
    }
