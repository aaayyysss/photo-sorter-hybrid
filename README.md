# photo-sorter-hybrid (Starter)

A hybrid architecture for your face-based photo sorter:

- **local-app/**: Runs on the user's PC. It loads local photos, computes **embeddings locally** (no big image uploads), and sends only small vectors and filenames to the cloud.
- **backend/**: Lightweight Flask API that holds registered **reference persons** (their mean embeddings) and performs sorting assignments by cosine similarity.
- **ui/**: Simple static dashboard to ping the backend and visualize basic info.
- **shared/**: API schema and notes.

## Quick Start

### 1) Backend (cloud / Oracle VM)
```bash
# On server
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# optional: copy config
cp config.example.json config.json
# Run (dev)
python app.py
# Or via gunicorn (recommended for prod)
# pip install gunicorn
# gunicorn -w 2 -b 0.0.0.0:8080 app:app
```

The backend listens by default on **0.0.0.0:8080** (edit inside `app.py`).

### 2) Local App (your PC)
```bash
cd local-app
python -m venv venv && source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
cp config.example.json config.json
# 2a) Build reference embeddings from local refs folder and register on server:
python send_refs.py --refs "/path/to/Refs"
# 2b) Compute inbox embeddings locally, get sorting from server, then MOVE/COPY locally:
python sort_local.py --inbox "/path/to/Inbox" --sorted "/path/to/Sorted" --mode move --threshold 0.32
```

> The local app never uploads heavy images. Only tiny embedding vectors (float32 arrays) and filenames go to the server.

### 3) UI (optional)
Open `ui/index.html` in a browser and point it to your backend URL (edit the base URL at the top of the file).

## Notes
- The local embedder is provided as a pluggable interface. By default, it uses **InsightFace** (CPU) if installed; otherwise it falls back to a **simple image hashing** (very rough) so you can test end-to-end without heavy models.
- The backend stores persons in-memory. You can easily swap to Redis/PostgreSQL later.
