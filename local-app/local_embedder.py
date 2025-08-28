import numpy as np

def _try_insightface():
    try:
        from insightface.app import FaceAnalysis
    except Exception:
        return None
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1)  # CPU
    def fn(img_bgr: np.ndarray):
        faces = app.get(img_bgr)
        if not faces:
            return None
        f = max(faces, key=lambda x: x.det_score)
        return f.embedding.astype(np.float32)
    return fn

def _try_imagehash():
    try:
        from PIL import Image
        import imagehash
    except Exception:
        return None
    def fn(img_bgr: np.ndarray):
        img_rgb = img_bgr[:, :, ::-1]
        pil = Image.fromarray(img_rgb)
        ph = imagehash.phash(pil)  # 64-bit hash
        bits = np.array([int(b) for b in bin(int(str(ph), 16))[2:].zfill(64)], dtype=np.float32)
        return bits
    return fn

def get_embedder():
    for maker in (_try_insightface, _try_imagehash):
        e = maker()
        if e:
            return e
    raise RuntimeError("No embedding backend available. Install insightface+onnxruntime or imagehash+Pillow.")

def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v) + eps
    return v / n

def embed_file(path: str, normalize=True):
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    fn = get_embedder()
    vec = fn(img)
    if vec is None:
        return None
    return l2_normalize(vec) if normalize else vec
