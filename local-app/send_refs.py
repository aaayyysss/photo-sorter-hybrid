import os, json, argparse, requests
from tqdm import tqdm
from local_embedder import embed_file

def walk_images(root):
    exts = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
    for r,_,files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                yield os.path.join(r,f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="Path to references root; subfolders = persons")
    ap.add_argument("--backend", default=None, help="Backend base URL, overrides config.json")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        cfg = json.load(open(args.config, "r", encoding="utf-8"))
    backend = args.backend or cfg.get("backend_url", "http://127.0.0.1:8080")

    persons = {}
    for name in sorted([d for d in os.listdir(args.refs) if os.path.isdir(os.path.join(args.refs, d))]):
        folder = os.path.join(args.refs, name)
        embs = []
        for img_path in tqdm(list(walk_images(folder)), desc=f"Embedding {name}"):
            vec = embed_file(img_path, normalize=True)
            if vec is not None:
                embs.append(vec.astype(float).tolist())
        if embs:
            persons[name] = embs

    payload = {"persons": [{"name": k, "embeddings": v} for k,v in persons.items()], "normalize": True}
    r = requests.post(f"{backend}/api/refs/register", json=payload, timeout=120)
    print("Server:", r.status_code, r.text)

if __name__ == "__main__":
    main()
