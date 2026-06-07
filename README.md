# 🛡️ Sentinel AI — Real-time Surveillance System

AI-powered surveillance dashboard with skeleton tracking, behaviour detection, and real-time alerts.

## Features
- 📷 Live camera (browser) OR 🎬 video upload
- 🦴 Skeleton pose drawing (MediaPipe)
- 🧍 Behaviour detection: Normal / Active / Fast Move
- ⚠️ Alert system: crowd, loitering, boundary, fast movement
- 📊 People count graph
- 🚗 Vehicle counting (cars + buses)

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Hugging Face Spaces
1. Create new Space → Streamlit SDK
2. Upload all files
3. Done ✅