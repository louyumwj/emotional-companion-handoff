"""检查压缩包内容：列出条目、大小，并预览文本/JSON 文件的结构。"""
import io
import json
import os
import sys
import zipfile

path = sys.argv[1]
preview_n = int(sys.argv[2]) if len(sys.argv) > 2 else 2

with zipfile.ZipFile(path) as z:
    infos = z.infolist()
    print(f"压缩包：{os.path.basename(path)}")
    print(f"条目数：{len(infos)}  解压后总大小：{sum(i.file_size for i in infos)/1e6:.2f} MB\n")
    print(f"{'大小':>12}  {'压缩后':>12}  名称")
    print("-" * 72)
    for i in infos:
        print(f"{i.file_size:>12,}  {i.compress_size:>12,}  {i.filename}")

    print("\n" + "=" * 72)
    print("文本类文件预览")
    print("=" * 72)
    shown = 0
    for i in infos:
        if i.is_dir() or i.file_size == 0:
            continue
        name = i.filename
        if not name.lower().endswith((".json", ".jsonl", ".txt", ".csv", ".md")):
            continue
        with z.open(i) as f:
            raw = f.read(min(i.file_size, 20000))
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", "ignore")

        print(f"\n---------- {name}  ({i.file_size:,} bytes) ----------")
        if name.lower().endswith(".json"):
            try:
                data = json.loads(text)
                print("顶层类型:", type(data).__name__,
                      "| 长度:", len(data) if hasattr(data, "__len__") else "-")
                sample = data[:preview_n] if isinstance(data, list) else data
                s = json.dumps(sample, ensure_ascii=False, indent=2)
                print(s[:2500])
            except Exception as exc:
                print("(JSON 解析失败，可能是被截断，显示前 1500 字符)", exc)
                print(text[:1500])
        else:
            lines = text.splitlines()
            print(f"行数(预览内): {len(lines)}")
            for line in lines[:8]:
                print("  ", line[:220])
        shown += 1
        if shown >= 6:
            print("\n(仅预览前 6 个文本文件)")
            break
