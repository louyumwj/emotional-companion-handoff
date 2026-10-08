"""下载 T1 数据基座所需的公开中文心理对话语料。

走 jsdelivr CDN 镜像（GitHub 直连在本机不通）。
注意 jsdelivr 对单文件有 20MB 上限，超限文件返回 403，脚本会记录并跳过。

用法：python tools/data/fetch_sources.py
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

CDN = "https://cdn.jsdelivr.net/gh/SmartFlowAI/EmoLLM@main/datasets"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

SOURCES = [
    ("data.json", "EmoLLM 自建通用多轮对话（心理健康）"),
    ("data_pro.json", "EmoLLM 自建增强版多轮对话"),
    ("multi_turn_dataset_1.json", "Smile 心理咨询多轮对话"),
    ("multi_turn_dataset_2.json", "CPsyCounD（ACL 2024 Findings）"),
    ("aiwei.json", "EmoLLM 角色扮演·温柔御姐（取样用）"),
    ("self_cognition_EmoLLM.json", "EmoLLM 自我认知 QA"),
]


def fetch(name: str, out_dir: str, retries: int = 3):
    dst = os.path.join(out_dir, name)
    if os.path.isfile(dst) and os.path.getsize(dst) > 1024:
        print(f"[skip] {name} 已存在 {os.path.getsize(dst)/1e6:.2f} MB")
        return "cached"
    url = f"{CDN}/{name}"
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as resp, open(dst, "wb") as f:
                total = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
            print(f"[ok]   {name}  {total/1e6:.2f} MB")
            return "ok"
        except urllib.error.HTTPError as exc:
            print(f"[http {exc.code}] {name}（jsdelivr 单文件上限 20MB，超限会 403）")
            if os.path.isfile(dst):
                os.remove(dst)
            return f"http{exc.code}"
        except Exception as exc:
            print(f"[retry {i+1}] {name}: {type(exc).__name__}: {exc}")
            time.sleep(2)
    return "failed"


def main():
    ws = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(ws, "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    print(f"输出目录：{out_dir}\n")
    results = {}
    for name, desc in SOURCES:
        print(f"--- {name}  ({desc})")
        results[name] = fetch(name, out_dir)
    print("\n===== 下载结果 =====")
    for name, status in results.items():
        size = ""
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            size = f"{os.path.getsize(p)/1e6:.2f} MB"
        print(f"  {name:32s} {status:10s} {size}")
    with open(os.path.join(out_dir, "_fetch_status.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
