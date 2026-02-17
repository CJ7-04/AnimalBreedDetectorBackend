from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import json
from pathlib import Path
from src.infer import BreedClassifier, load_image

app = FastAPI(title="BPA Breed ID Service", version="0.3.0")
clf = None

# -------------------------
# RESPONSE MODEL
# -------------------------
class PredictResponse(BaseModel):
    topk: List[dict]
    suggestion: Optional[str] = None
    breed_info: Optional[dict] = None
    confidence_message: Optional[str] = None   # ✅ NEW

# -------------------------
# LOAD MODEL
# -------------------------
@app.on_event("startup")
def load_model():
    global clf
    try:
        clf = BreedClassifier()
    except Exception as e:
        print("Model load failed:", e)
        clf = None

# -------------------------
# BREED DATABASE HELPERS
# -------------------------
def load_db():
    return json.loads(Path("src/breed_db.json").read_text(encoding="utf-8"))

def get_breed_info(breed_name):
    db = load_db()
    for b in db["breeds"]:
        if b["name"].lower() == breed_name.lower():
            return b
    return None

# -------------------------
# CONFIDENCE GUIDANCE
# -------------------------
def confidence_guidance(conf):
    if conf >= 85:
        return {
            "en": "High confidence — result is reliable.",
            "hi": "उच्च विश्वसनीयता — परिणाम सही है।"
        }
    elif conf >= 60:
        return {
            "en": "Medium confidence — please verify manually.",
            "hi": "मध्यम विश्वसनीयता — कृपया जांच करें।"
        }
    else:
        return {
            "en": "Low confidence — take a clearer photo and try again.",
            "hi": "कम विश्वसनीयता — साफ फोटो लेकर फिर प्रयास करें।"
        }

# -------------------------
# BREEDS LIST
# -------------------------
@app.get("/breeds")
def get_breeds():
    db = load_db()
    return {i: b["name"] for i, b in enumerate(db["breeds"])}

# -------------------------
# PREDICT ENDPOINT
# -------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.6),
    topk: int = Form(3),
    lang: str = Form("en")
):
    if clf is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded. Train first."})

    img_bytes = await file.read()
    img = load_image(img_bytes)

    preds = clf.predict(img, topk=topk)

    # convert confidence to %
    for p in preds:
        p["confidence"] = round(p["confidence"] * 100, 2)

    best = preds[0] if preds else None
    best_conf = best["confidence"] if best else 0

    suggestion = best["breed"] if best and best_conf >= threshold * 100 else None

    breed_info = get_breed_info(best["breed"]) if best else None

    # 🌐 Language & display handling
    if breed_info:
        if lang == "hi":
            breed_info["display_name"] = breed_info.get("local_names", {}).get("hi", breed_info["name"])
            breed_info["farmer_tip"] = breed_info.get("farmer_tips", {}).get("hi")
        else:
            breed_info["display_name"] = breed_info["name"]
            breed_info["farmer_tip"] = breed_info.get("farmer_tips", {}).get("en")

    # 🌾 Confidence guidance
    message = confidence_guidance(best_conf)
    confidence_message = message["hi"] if lang == "hi" else message["en"]

    return {
        "topk": preds,
        "suggestion": suggestion,
        "breed_info": breed_info,
        "confidence_message": confidence_message
    }

# -------------------------
# BPA HOOK (UNCHANGED)
# -------------------------
@app.post("/bpa_hook")
async def bpa_hook(
    bpa_breed: str = Form(...),
    file: UploadFile = File(...),
    threshold: float = Form(0.6)
):
    if clf is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})

    img_bytes = await file.read()
    img = load_image(img_bytes)

    preds = clf.predict(img, topk=3)

    ai_suggestion = preds[0]["breed"] if preds and preds[0]["confidence"] >= threshold else None

    action = "confirm" if ai_suggestion == bpa_breed else (
        "override" if ai_suggestion else "manual_review"
    )

    return {
        "bpa_breed": bpa_breed,
        "ai_top3": preds,
        "ai_suggestion": ai_suggestion,
        "action": action
    }
