"""通过 hf-mirror 下载模型/数据集（本机 GitHub 与 huggingface.co 直连不通）。"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from huggingface_hub import snapshot_download


def main():
    repo = sys.argv[1]
    local = sys.argv[2]
    patterns = sys.argv[3].split(",") if len(sys.argv) > 3 else [
        "*.json", "*.model", "*.safetensors", "*.txt"]
    os.makedirs(local, exist_ok=True)
    print(f"repo={repo}\nlocal={local}\nendpoint={os.environ['HF_ENDPOINT']}", flush=True)
    path = snapshot_download(repo_id=repo, local_dir=local, allow_patterns=patterns,
                             max_workers=4)
    total = 0
    for root, _, files in os.walk(local):
        for f in files:
            if ".cache" in root:
                continue
            p = os.path.join(root, f)
            total += os.path.getsize(p)
            print(f"  {os.path.relpath(p, local):55s} {os.path.getsize(p)/1e6:9.2f} MB")
    print(f"合计 {total/1e6:.1f} MB  ->  {path}")


if __name__ == "__main__":
    main()
