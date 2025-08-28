import os
import json
from typing import List, Dict, Any
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# In-memory store: person_name -> dict(mean: np.ndarray, count: int)
PERSONS: Dict[str, Dict[str, Any]] = {}

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8080"))

def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v) + eps
    return v / n

def mean_embedding(embs: List[np.ndarray], normalize=True) -> np.ndarray:
    if not embs:
        raise ValueError("empty embedding list")
    M = np.vstack(embs)
    m = M.mean(axis=0)
    return l2_normalize(m) if normalize else m

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "persons": list(PERSONS.keys())})

@app.post("/api/refs/register")
def register_refs():
    """
    Body: { "persons": [{"name": "...","embeddings":[[...],[...]]}, ...], "normalize": true }
    """
    data = request.get_json(silent=True) or {}
    persons = data.get("persons", [])
    normalize = bool(data.get("normalize", True))
    registered = []
    try:
        for p in persons:
            name = str(p.get("name", "")).strip()
            emb_list = p.get("embeddings", [])
            if not name or not emb_list:
                continue
            embs = [np.array(vec, dtype=np.float32) for vec in emb_list]
            M = np.vstack(embs)
            m = M.mean(axis=0)
            if normalize:
                n = np.linalg.norm(m) + 1e-12
                m = m / n
            PERSONS[name] = {"mean": m, "count": len(embs)}
            registered.append(name)
        return jsonify({"status": "ok", "registered": registered, "total_persons": len(PERSONS)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.post("/api/sort")
def sort_embeddings():
    """
    Body: { "inbox": [{"file":"...", "embedding":[...]}], "threshold": 0.32, "multi_label": false }
    """
    data = request.get_json(silent=True) or {}
    inbox = data.get("inbox", [])
    threshold = float(data.get("threshold", 0.32))
    multi_label = bool(data.get("multi_label", False))

    # Prepare person matrix
    names = list(PERSONS.keys())
    if not names:
        return jsonify({"status": "error", "message": "no persons registered"}), 400
    import numpy as np
    P = np.vstack([PERSONS[n]["mean"] for n in names])
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)

    assignments = []
    for item in inbox:
        fname = item.get("file", "")
        vec_list = item.get("embedding", [])
        vec = np.array(vec_list, dtype=np.float32)
        if vec.size == 0:
            assignments.append({"file": fname, "best": None, "all": []})
            continue
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        scores = (P @ vec).astype(np.float32)
        all_scores = [{"person": names[i], "score": float(scores[i])} for i in range(len(names))]
        all_scores.sort(key=lambda x: x["score"], reverse=True)

        best = None
        if all_scores and all_scores[0]["score"] >= threshold:
            best = {"person": all_scores[0]["person"], "score": all_scores[0]["score"]}

        if multi_label:
            kept = [s for s in all_scores if s["score"] >= threshold]
            assignments.append({"file": fname, "best": best, "all": kept})
        else:
            assignments.append({"file": fname, "best": best, "all": all_scores})

    return jsonify({"status": "ok", "assignments": assignments, "persons": names, "threshold": threshold})

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
