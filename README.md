# BPA Breed Identification (Prototype)

AI-driven breed recognition for cattle and buffaloes to assist Field Level Workers (FLWs) during registration on the Bharat Pashudhan App (BPA).  
This repo includes: model training (PyTorch), an inference service (FastAPI), and a simple FLW UI (Streamlit).

## Features
- Transfer learning on EfficientNet/ResNet with strong augmentations for background/pose/lighting variability
- Top-k predictions with confidence and thresholded suggestions for decision support
- Lightweight local inference (CPU) + optional ONNX export
- Breed database with aliases and metadata (JSON) for standardized names
- REST API for easy integration with BPA
- Streamlit front-end for FLWs (camera/upload, offline-capable local inference option)

## Quick Start

<!-- Here’s a complete README.md for your project including all commands to run backend and frontend:

# BPA Breed Identification Project

This project identifies cow breeds using a trained EfficientNet model with a FastAPI backend and Streamlit frontend.

---

## Project Structure



project-root/
│
├─ app/
│ └─ frontend_streamlit.py # Streamlit frontend
│
├─ models/
│ ├─ model.pth # Trained PyTorch model
│ └─ labels.json # Label mapping
│
├─ src/
│ └─ backend.py # FastAPI backend
│
├─ .venv/ # Python virtual environment
├─ start_backend.bat # Start backend server (Windows)
├─ start_frontend.bat # Start Streamlit frontend (Windows)
└─ README.md


---

## Setup Instructions

1. **Clone the project**

```bash
git clone <repo-url>
cd project-root


Create virtual environment (if not already created)

python -m venv .venv


Activate virtual environment

Windows:

.venv\Scripts\activate


Linux/macOS:

source .venv/bin/activate


Install dependencies

pip install -r requirements.txt

Running the Backend (FastAPI)

Windows:

start_backend.bat


Or manually:

call .venv\Scripts\activate
uvicorn src.backend:app --reload --host 0.0.0.0 --port 8000


Linux/macOS:

chmod +x start.sh
./start.sh


Backend will run at: http://localhost:8000

Running the Frontend (Streamlit)

Windows:

start_frontend.bat


Or manually:

call .venv\Scripts\activate
streamlit run app/frontend_streamlit.py


Frontend will open in your browser and communicate with the backend.

API Endpoints

GET / - Check backend status

POST /predict - Upload an image to get predicted breed

GET /breeds - Get list of available breeds

Notes

Make sure backend is running before using frontend.

Ensure models/model.pth and models/labels.json exist.

Streamlit frontend automatically fetches breeds and predicts breed for uploaded images.

Troubleshooting

FileNotFoundError for model: Make sure the model is at models/model.pth.

Streamlit always shows same breed: Make sure backend loads correct model and labels.

Port issues: Change --port 8000 in uvicorn if already in use. -->