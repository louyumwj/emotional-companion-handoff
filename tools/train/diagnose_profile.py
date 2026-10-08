"""T2 画像头诊断：判断是「信号不足」还是「阈值/校准问题」。

输出每个标签的：
  prevalence（正例率）、AUC（排序能力）、最佳可达 F1（对该标签单独扫阈值的上界）
判据：
  AUC ≥ 0.75 且最佳 F1 明显 > 0  → 信号在，是校准问题（加 pos_weight / 调阈值即可）
  AUC ≈ 0.5                      → 信号不在，需要加长上下文 / 提高 loss 权重 / 换输入构造
另外计算「永远输出最高频标签组合」的基线，看是否值得用。
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "train"))
from labels import INTERESTS, STYLES, TRAITS  # noqa: E402
from train_multitask import MultitaskDataset, MultiTaskModel  # noqa: E402
from model_paths import resolve_base  # noqa: E402

GROUPS = [("personality_traits", TRAITS), ("interests", INTERESTS), ("style", STYLES)]


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def best_f1(y, s):
    order = np.argsort(-s)
    y = np.asarray(y)[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    fn = tp[-1] - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = tp / np.maximum(tp + fp, 1e-9)
        rec = tp / np.maximum(tp + fn, 1e-9)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(np.nanmax(f1)) if len(f1) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("legacy_threads", nargs="?", type=int)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--base", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.legacy_threads or args.threads)
    ckpt = torch.load(os.path.join(ROOT, "models", "t2_multitask", "best.pt"),
                      map_location="cpu", weights_only=True)
    base = resolve_base(ROOT, ckpt["base"], args.base)
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = MultiTaskModel(base)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "processed", "views",
                                                     "val_cls.jsonl"), encoding="utf-8")]
    dl = DataLoader(MultitaskDataset(rows, tok, ckpt.get("max_length", 256)),
                    batch_size=32, shuffle=False)
    probs = {name: [] for name, _ in GROUPS}
    with torch.no_grad():
        for batch in dl:
            logits = model(batch["input_ids"], batch["attention_mask"])
            for name, _ in GROUPS:
                probs[name].append(torch.sigmoid(logits[name].float()).numpy())
    probs = {k: np.concatenate(v, 0) for k, v in probs.items()}

    print(f"{'字段/标签':40s}{'正例率':>9}{'AUC':>8}{'最佳F1':>9}{'均值概率':>10}")
    print("-" * 78)
    summary = {}
    for name, labels in GROUPS:
        aucs, f1s = [], []
        for j, lab in enumerate(labels):
            y = np.array([1 if lab in r["user_profile"].get(name, []) else 0 for r in rows])
            a = auc(y, probs[name][:, j])
            f = best_f1(y, probs[name][:, j])
            aucs.append(a if a is not None else float("nan"))
            f1s.append(f)
            if y.sum() > 0:
                print(f"{name+'/'+lab:40s}{y.mean():>9.3f}"
                      f"{(a if a is not None else float('nan')):>8.3f}{f:>9.3f}"
                      f"{probs[name][:,j].mean():>10.3f}")
        summary[name] = {"mean_auc": float(np.nanmean(aucs)),
                         "mean_best_f1": float(np.mean(f1s))}
        print(f"  >> {name} 平均 AUC={summary[name]['mean_auc']:.3f}  "
              f"平均最佳F1={summary[name]['mean_best_f1']:.3f}")

    # 基线：永远输出"最高频标签组合"
    print("\n=== 基线对照 ===")
    for name, labels in GROUPS:
        cnt = collections.Counter()
        for r in rows:
            cnt[tuple(sorted(r["user_profile"].get(name, [])))] += 1
        top, c = cnt.most_common(1)[0]
        print(f"{name:22s} 最高频组合={list(top)}  占比={c/len(rows):.3f}")

    with open(os.path.join(ROOT, "reports", "t2_diagnosis.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
