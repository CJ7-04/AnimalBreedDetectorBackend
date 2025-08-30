from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import json
from pathlib import Path
from PIL import Image
from src.infer import BreedClassifier, load_image


app = FastAPI(title="BPA Breed ID Service", version="0.1.0")
clf = None

class PredictResponse(BaseModel):
    topk: List[dict]
    suggestion: Optional[str] = None

@app.on_event("startup")
def load_model():
    global clf
    try:
        clf = BreedClassifier()
    except Exception as e:
        print("Model load failed:", e)
        clf = None

@app.get("/breeds")
def get_breeds():
    db = json.loads(Path("src/breed_db.json").read_text())
    return db

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), threshold: float = Form(0.6), topk: int = Form(3)):
    if clf is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded. Train first."})
    img_bytes = await file.read()
    img = load_image(img_bytes)
    preds = clf.predict(img, topk=topk)
    suggestion = preds[0]["breed"] if preds and preds[0]["confidence"] >= threshold else None
    return {"topk": preds, "suggestion": suggestion}

@app.post("/bpa_hook")
async def bpa_hook(bpa_breed: str = Form(...), file: UploadFile = File(...), threshold: float = Form(0.6)):
    if clf is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})
    img_bytes = await file.read()
    img = load_image(img_bytes)
    preds = clf.predict(img, topk=3)
    ai_suggestion = preds[0]["breed"] if preds and preds[0]["confidence"] >= threshold else None
    action = "confirm" if ai_suggestion == bpa_breed else ( "override" if ai_suggestion else "manual_review")
    return {
        "bpa_breed": bpa_breed,
        "ai_top3": preds,
        "ai_suggestion": ai_suggestion,
        "action": action
    }
