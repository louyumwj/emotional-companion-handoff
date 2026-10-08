"""从 ModelScope 下载模型文件（本机 hf-mirror 的 LFS 走 xethub 不通）。

用法：
  python tools/train/download_ms.py Qwen/Qwen2.5-0.5B-Instruct models/Qwen2.5-0.5B-Instruct \
      config.json,generation_config.json,merges.txt,tokenizer.json,tokenizer_config.json,vocab.json,model.safetensors
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.parse

API = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={f}"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def download(repo, rel, dst, retries=3):
    url = API.format(repo=repo, f=urllib.parse.quote(rel))
    temporary = dst + ".part"
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as r, open(temporary, "wb") as f:
                expected = int(r.headers.get("Content-Length", 0))
                total = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
                    if total % (50 << 20) < (1 << 20):
                        print(f"    ... {total/1e6:.0f} MB", flush=True)
            if total == 0 or (expected and total != expected):
                raise IOError(f"Incomplete download: received {total}, expected {expected}")
            if rel.endswith(".json"):
                with open(temporary, encoding="utf-8") as f:
                    json.load(f)
            os.replace(temporary, dst)
            return total
        except Exception as exc:
            print(f"    [重试 {i+1}] {rel}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(2)
    if os.path.isfile(temporary):
        os.remove(temporary)
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("local")
    ap.add_argument("files", help="Comma-separated required files")
    ap.add_argument("--optional-files", default="")
    args = ap.parse_args()
    repo, local = args.repo, args.local
    required = [f for f in args.files.split(",") if f]
    optional = [f for f in args.optional_files.split(",") if f and f not in required]
    files = required + optional
    os.makedirs(local, exist_ok=True)
    print(f"ModelScope 仓库 {repo} -> {local}", flush=True)
    ok, fail = [], []
    for rel in files:
        dst = os.path.join(local, rel)
        if os.path.commonpath([os.path.abspath(local), os.path.abspath(dst)]) != os.path.abspath(local):
            ap.error("File paths must stay inside the model directory")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        n = download(repo, rel, dst)
        size = os.path.getsize(dst) if os.path.isfile(dst) else 0
        if n < 0 or size == 0:
            print(f"  [失败] {rel}", flush=True)
            if rel in required:
                fail.append(rel)
            else:
                print("    可选文件，继续")
        else:
            print(f"  [成功] {rel:32s} {size/1e6:9.2f} MB", flush=True)
            ok.append(rel)
    print(f"\n成功 {len(ok)} 个，失败 {len(fail)} 个")
    if fail:
        print("失败清单:", fail)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
