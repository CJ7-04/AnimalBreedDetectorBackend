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
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
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

    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"], strict=False)
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

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
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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
        probs = torch.softmax(outputs, dim=1)  # convert logits to probabilities
        predicted_idx = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, predicted_idx])

    class_name = LABELS.get(str(predicted_idx), "Unknown")

    return {
        "predicted_class": class_name,
        "confidence": round(confidence, 3)  # optional, useful for frontend
    }

@app.get("/breeds")
def get_breeds():
    if LABELS:
        return LABELS
    return {"error": "Labels file not found or empty"}

