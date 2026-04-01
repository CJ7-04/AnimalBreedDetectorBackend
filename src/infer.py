import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import json
import os

# -------------------------
# Model definition
# -------------------------
class BreedClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# -------------------------
# Model loader
# -------------------------
def load_model(checkpoint_path, num_classes, device="cpu"):
    model = BreedClassifier(num_classes)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # ✅ STRICT loading (IMPORTANT)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model.to(device)

# -------------------------
# Image loader
# -------------------------
def load_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB") #image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust to your frontend domain for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cpu"
MODEL_PATH = "models/model.pth"
LABELS_PATH = "models/labels.json"

# Load labels
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        LABELS = json.load(f)

    LABELS = {int(k): v for k, v in LABELS.items()}

else:
    LABELS = {}

NUM_CLASSES = len(LABELS)
model = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"message": "Backend is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    tensor = load_image(image_bytes).to(DEVICE)

    with torch.no_grad():
     outputs = model(tensor)
     probs = torch.softmax(outputs, dim=1)

    # Top-3 predictions
    top_probs, top_idxs = torch.topk(probs, 3)

    results = []
    for i in range(3):
        idx = int(top_idxs[0][i])
        conf = float(top_probs[0][i])

        results.append({
            "breed": LABELS.get(idx, "Unknown"),
            "confidence": round(conf, 3)
        })

    return {
    "top_predictions": results
}
@app.get("/breeds")
def get_breeds():
    if LABELS:
        return LABELS
    return {"error": "Labels file not found or empty"}
