from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
from pathlib import Path
import json, torch
from src.infer import load_image, model, LABELS

app = FastAPI(title="Breed ID Service", version="1.0")

# -------------------------
# Load breed database
# -------------------------
def load_db():
    return json.loads(Path("src/breed_db.json").read_text(encoding="utf-8"))

def get_breed_info(name):
    db = load_db()
    for b in db["breeds"]:
        if b["name"].lower() == name.lower():
            return b
    return None

# -------------------------
# Confidence guidance
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
def breeds():
    return LABELS

# -------------------------
# PREDICT
# -------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.4),
    topk: int = Form(3),
    lang: str = Form("en")
):
    try:
        image_bytes = await file.read()
        tensor = load_image(image_bytes)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        # top predictions
        top_probs, top_idxs = torch.topk(probs, topk)

        preds = []
        for p, idx in zip(top_probs, top_idxs):
            breed = LABELS[str(idx.item())]
            preds.append({
                "breed": breed,
                "confidence": round(float(p.item()) * 100, 2)
            })

        best = preds[0]
        suggestion = best["breed"] if best["confidence"] >= threshold*100 else None

        breed_info = get_breed_info(best["breed"])

        if breed_info:
            if lang == "hi":
                breed_info["display_name"] = breed_info["local_names"]["hi"]
                breed_info["farmer_tip"] = breed_info["farmer_tips"]["hi"]
            else:
                breed_info["display_name"] = breed_info["name"]
                breed_info["farmer_tip"] = breed_info["farmer_tips"]["en"]

        message = confidence_guidance(best["confidence"])
        confidence_message = message["hi"] if lang == "hi" else message["en"]

        return {
            "topk": preds,
            "suggestion": suggestion,
            "breed_info": breed_info,
            "confidence_message": confidence_message
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
