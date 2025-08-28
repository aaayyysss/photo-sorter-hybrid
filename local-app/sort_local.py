import os, json, argparse, shutil, requests
from tqdm import tqdm
from local_embedder import embed_file

def walk_images(root):
    exts = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
    for r,_,files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                yield os.path.join(r,f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", required=True, help="Path to inbox folder")
    ap.add_argument("--sorted", required=True, help="Path to SORTED output root (local)")
    ap.add_argument("--backend", default=None, help="Backend base URL, overrides config.json")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--mode", default=None, choices=["move","copy","link"])
    args = ap.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        cfg = json.load(open(args.config, "r", encoding="utf-8"))
    backend = args.backend or cfg.get("backend_url", "http://127.0.0.1:8080")
    threshold = args.threshold if args.threshold is not None else cfg.get("threshold", 0.32)
    mode = args.mode or cfg.get("mode", "move")

    inbox_items = []
    paths = list(walk_images(args.inbox))
    for p in tqdm(paths, desc="Embedding inbox"):
        vec = embed_file(p, normalize=True)
        if vec is None:
            continue
        inbox_items.append({"file": os.path.relpath(p, args.inbox), "embedding": vec.tolist()})

    payload = {"inbox": inbox_items, "threshold": float(threshold), "multi_label": False}
    r = requests.post(f"{backend}/api/sort", json=payload, timeout=300)
    data = r.json()
    if data.get("status") != "ok":
        print("Server error:", data)
        return

    for a in data["assignments"]:
        src_abs = os.path.join(args.inbox, a["file"])
        best = a.get("best")
        if not best:
            dst_abs = os.path.join(args.sorted, "_unassigned", os.path.basename(a["file"]))
        else:
            person = best["person"]
            dst_abs = os.path.join(args.sorted, person, os.path.basename(a["file"]))
        ensure_dir(os.path.dirname(dst_abs))
        if mode == "move":
            shutil.move(src_abs, dst_abs)
        elif mode == "copy":
            shutil.copy2(src_abs, dst_abs)
        elif mode == "link":
            try:
                os.link(src_abs, dst_abs)
            except Exception:
                shutil.copy2(src_abs, dst_abs)

    print(f"Done. Output at: {args.sorted}")

if __name__ == "__main__":
    main()
