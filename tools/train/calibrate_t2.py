"""T2 后处理：类别先验校正 + 多标签阈值校准 + 官方格式预测。

方法学要点：按会话分组随机切分 calib / holdout，阈值只在 calib 上调。
旧检查点曾用整个 val 选优，holdout 指标仅用于诊断；后续训练也必须保留同一 holdout。

产出：
  models/t2_multitask/calib.json          先验温度 + 每组逐标签阈值
  reports/t2_pred_val.jsonl               官方格式预测（四字段，memory_refs 暂空）
  reports/T2_报告.md                      指标报告
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
sys.path.insert(0, os.path.join(ROOT, "tools", "eval"))
from labels import EMOTIONS, INTERESTS, STYLES, TRAITS  # noqa: E402
from train_multitask import MultitaskDataset, MultiTaskModel  # noqa: E402
from model_paths import resolve_base  # noqa: E402
from validation_split import split_validation  # noqa: E402

GROUPS = [("personality_traits", TRAITS), ("interests", INTERESTS), ("style", STYLES)]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    E, P = [], []
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        logits = model(b["input_ids"], b["attention_mask"])
        E.append(torch.softmax(logits["emotion"].float(), -1).cpu().numpy())
        P.append({name: torch.sigmoid(logits[name].float()).cpu().numpy()
                  for name, _ in GROUPS})
    emo = np.concatenate(E, 0)
    prof = {name: np.concatenate([p[name] for p in P], 0) for name, _ in GROUPS}
    return emo, prof


def emotion_metrics(probs, gold_ids):
    pred = probs.argmax(1)
    acc = float((pred == np.array(gold_ids)).mean())
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
    for g, p in zip(gold_ids, pred.tolist()):
        if g == p:
            tp[EMOTIONS[g]] += 1
        else:
            fp[EMOTIONS[p]] += 1
            fn[EMOTIONS[g]] += 1
    f1s = [prf(tp[e], fp[e], fn[e])[2] for e in EMOTIONS]
    return acc, float(np.mean(f1s)), pred


def multilabel_f1(prob, gold_sets, labels, thresholds):
    tp = fp = fn = 0
    exact = 0
    for i, gs in enumerate(gold_sets):
        ps = {labels[j] for j in range(len(labels)) if prob[i, j] >= thresholds[j]}
        tp += len(ps & gs); fp += len(ps - gs); fn += len(gs - ps)
        if ps == gs:
            exact += 1
    p, r, f = prf(tp, fp, fn)
    return p, r, f, exact / max(1, len(gold_sets))


def tune_thresholds(prob, gold_sets, labels, grid=None):
    grid = grid or [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    best = []
    for j, _ in enumerate(labels):
        scores = []
        for t in grid:
            tp = fp = fn = 0
            for i, gs in enumerate(gold_sets):
                hit = prob[i, j] >= t
                ing = labels[j] in gs
                tp += hit and ing
                fp += hit and not ing
                fn += (not hit) and ing
            scores.append((prf(tp, fp, fn)[2], t))
        scores.sort(reverse=True)
        best.append(scores[0][1])
    return best


def tune_global_for_exact(prof, gold_by_group, idx, grid=None):
    """全局单阈值，目标 = 三组数组**完全一致率**（保守工作点）。"""
    grid = grid or [round(float(x), 2) for x in np.arange(0.30, 1.01, 0.05)] + [1.01]
    best_t, best_rate = 1.01, -1.0
    rates = []
    for t in grid:
        exact = 0
        for i in idx:
            ok = True
            for name, labels in GROUPS:
                gs = gold_by_group[name][i]
                ps = {labels[j] for j in range(len(labels)) if prof[name][i, j] >= t}
                if ps != gs:
                    ok = False
                    break
            exact += ok
        rate = exact / max(1, len(idx))
        rates.append((round(float(t), 2), round(rate, 4)))
        if rate > best_rate:
            best_rate, best_t = rate, t
    return best_t, best_rate, rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/t2_multitask/best.pt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--base", default="", help="Override a legacy checkpoint's base model path")
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    ckpt_path = os.path.join(ROOT, args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    base = resolve_base(ROOT, ckpt["base"], args.base)
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = MultiTaskModel(base)
    model.load_state_dict(ckpt["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "processed", "views",
                                                     "val_cls.jsonl"), encoding="utf-8")]
    ds = MultitaskDataset(rows, tok, ckpt.get("max_length", 384))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    emo, prof = predict(model, dl, device)
    gold_ids = [EMOTIONS.index(r["emotion_label"]) for r in rows]

    n = len(rows)
    idx_calib, idx_test = split_validation(rows, args.split_seed)
    gold_calib = [gold_ids[i] for i in idx_calib]
    gold_test = [gold_ids[i] for i in idx_test]
    selection_reserved = (ckpt.get("selection_split") == "conversation_calib"
                          and ckpt.get("split_seed") == args.split_seed)
    empty_test = sum(not any(rows[i]["user_profile"].get(name, []) for name, _ in GROUPS)
                     for i in idx_test) / len(idx_test)

    # ---- 1. 情绪：先验温度校正（在 calib 上调 tau） ----
    prior = np.bincount(gold_calib, minlength=len(EMOTIONS)).astype(float)
    prior = np.maximum(prior, 1.0); prior /= prior.sum()
    best_tau, best_calib = 0.0, -1.0
    for tau in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        adj = emo[idx_calib] / (prior ** tau)
        _, mf1, _ = emotion_metrics(adj, gold_calib)
        if mf1 > best_calib:
            best_calib, best_tau = mf1, tau
    adj_all = emo / (prior ** best_tau)
    acc_c, mf1_c, _ = emotion_metrics(adj_all[idx_calib], gold_calib)
    acc_t, mf1_t, pred_test = emotion_metrics(adj_all[idx_test], gold_test)
    acc_f, mf1_f, pred_all = emotion_metrics(adj_all, gold_ids)

    # ---- 2. 画像：逐标签阈值（calib 调，test 报） ----
    th = {}
    prof_report = {}
    for name, labels in GROUPS:
        gold_sets = [set(r["user_profile"].get(name, [])) for r in rows]
        calib_sets = [gold_sets[i] for i in idx_calib]
        test_sets = [gold_sets[i] for i in idx_test]
        th[name] = tune_thresholds(prof[name][idx_calib], calib_sets, labels)
        r_c = multilabel_f1(prof[name][idx_calib], calib_sets, labels, th[name])
        r_t = multilabel_f1(prof[name][idx_test], test_sets, labels, th[name])
        # 全局单阈值对照
        g0 = [0.5] * len(labels)
        r_g = multilabel_f1(prof[name][idx_test], test_sets, labels, g0)
        prof_report[name] = {"calib": r_c, "test_tuned": r_t, "test_0.5": r_g,
                             "thresholds": th[name]}

    # 三组联合"完全一致率"（在 test 上）
    exact_test = 0
    for i in idx_test:
        ok = True
        for name, labels in GROUPS:
            gs = set(rows[i]["user_profile"].get(name, []))
            ps = {labels[j] for j in range(len(labels)) if prof[name][i, j] >= th[name][j]}
            if ps != gs:
                ok = False
        exact_test += ok
    exact_test /= len(idx_test)

    # ---- 工作点 B：全局阈值，目标 = 全对率（极端保守；t>1 等价于"全空"） ----
    gold_by_group = {name: [set(r["user_profile"].get(name, [])) for r in rows]
                     for name, _ in GROUPS}
    gb_t, gb_rate, gb_curve = tune_global_for_exact(prof, gold_by_group, idx_calib)
    # 在 test 上评估工作点 B
    exact_b_test = 0
    f1_b = {}
    for name, labels in GROUPS:
        r = multilabel_f1(prof[name][idx_test], [gold_by_group[name][i] for i in idx_test], labels,
                          [gb_t] * len(labels))
        f1_b[name] = r
    for i in idx_test:
        ok = True
        for name, labels in GROUPS:
            ps = {labels[j] for j in range(len(labels)) if prof[name][i, j] >= gb_t}
            if ps != gold_by_group[name][i]:
                ok = False
                break
        exact_b_test += ok
    exact_b_test /= len(idx_test)

    # ---- 输出 ----
    outdir = os.path.join(ROOT, "models", "t2_multitask")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(outdir, "calib.json"), "w", encoding="utf-8") as f:
        json.dump({"emotion_prior_tau": best_tau, "class_prior": prior.tolist(),
                   "thresholds": th,
                   "exact_mode_global_threshold": gb_t,
                   "exact_mode_calib_rate": gb_rate,
                   "split_method": "conversation_random", "split_seed": args.split_seed,
                   "calib_instance_ids": [rows[i]["instance_id"] for i in idx_calib],
                   "holdout_instance_ids": [rows[i]["instance_id"] for i in idx_test]},
                  f, ensure_ascii=False, indent=2)

    # 工作点 A：逐标签阈值（优化分字段 F1）
    pred_path = os.path.join(ROOT, "reports", "t2_pred_val.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            prof_out = {}
            for name, labels in GROUPS:
                prof_out[name] = [labels[j] for j in range(len(labels))
                                  if prof[name][i, j] >= th[name][j]]
            f.write(json.dumps({
                "instance_id": r["instance_id"],
                "response_text": "",
                "emotion_label": EMOTIONS[int(pred_all[i])],
                "user_profile": prof_out,
                "memory_refs": [],
            }, ensure_ascii=False) + "\n")

    # 工作点 B：全局阈值（优化三组全对率，如果官方按"完全一致"计分就用这个）
    pred_b = os.path.join(ROOT, "reports", "t2_pred_val_exactmode.jsonl")
    with open(pred_b, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            prof_out = {}
            for name, labels in GROUPS:
                prof_out[name] = [labels[j] for j in range(len(labels))
                                  if prof[name][i, j] >= gb_t]
            f.write(json.dumps({
                "instance_id": r["instance_id"],
                "response_text": "",
                "emotion_label": EMOTIONS[int(pred_all[i])],
                "user_profile": prof_out,
                "memory_refs": [],
            }, ensure_ascii=False) + "\n")

    L = ["# T2 结构化预测头 · 报告", "",
         f"- 检查点：`{args.ckpt}`",
         f"- 验证实例：{n}（calib {len(idx_calib)} / holdout {len(idx_test)}）",
         f"- 划分：按会话分组随机切分，seed={args.split_seed}，两组会话无交集",
         "- 选优隔离：" + ("训练使用同一 calib 选检查点，holdout 保留"
                            if selection_reserved else "旧检查点曾在全 val 上选优；本次 holdout 仅作诊断，不是独立测试结果"), "",
         "## 1. 情绪（16 分类）", "",
         f"- 先验校正温度 tau = **{best_tau}**", "",
         "| 数据集 | Accuracy | Macro-F1 |", "|---|---|---|",
         f"| calib（调参用） | {acc_c:.4f} | {mf1_c:.4f} |",
         f"| **holdout** | **{acc_t:.4f}** | **{mf1_t:.4f}** |",
         f"| 全 val | {acc_f:.4f} | {mf1_f:.4f} |", "",
         "历史全 val 基线（不与 holdout 直接比较）：多数类 Acc 0.3190 / Macro-F1 0.0302；规则层 Acc 0.2587 / Macro-F1 0.1061", "",
         "## 2. 用户画像（多标签，阈值在校准集上搜）", "",
         "| 字段 | holdout P | holdout R | holdout F1 | holdout 全对率 | 0.5 阈值下 F1 |",
         "|---|---|---|---|---|---|"]
    for name, labels in GROUPS:
        r = prof_report[name]
        L.append(f"| {name} | {r['test_tuned'][0]:.3f} | {r['test_tuned'][1]:.3f} | "
                 f"**{r['test_tuned'][2]:.3f}** | {r['test_tuned'][3]:.3f} | "
                 f"{r['test_0.5'][2]:.3f} |")
    L += [f"| **三组完全一致率** | | | | **{exact_test:.4f}** | |", "",
          f"同一 holdout 对照：全空画像全对率 {empty_test:.4f}（三组 F1 均为 0）", "",
          "### 两个工作点（官方指标未知，两个都必须给出）", "",
          "| 工作点 | 全局阈值 | 三组全对率(holdout) | traits F1 | interests F1 | style F1 | 适用场景 |",
          "|---|---|---|---|---|---|---|",
          f"| **A 逐标签阈值** | 每标签不同 | {exact_test:.4f} | "
          f"{prof_report['personality_traits']['test_tuned'][2]:.3f} | "
          f"{prof_report['interests']['test_tuned'][2]:.3f} | "
          f"{prof_report['style']['test_tuned'][2]:.3f} | 官方按分字段 F1 计分 |",
          f"| **B 全局阈值** | {gb_t} | **{exact_b_test:.4f}** | "
          f"{f1_b['personality_traits'][2]:.3f} | {f1_b['interests'][2]:.3f} | "
          f"{f1_b['style'][2]:.3f} | 官方按三组完全一致计分 |",
          f"| 参考：全空画像 | >1.0 | {empty_test:.4f} | 0 | 0 | 0 | 同一 holdout 保守兜底 |", "",
          f"工作点 B 全对率随全局阈值的变化（calib）：{gb_curve}", "",
          "> **核心权衡**：A 优化分字段 F1，B 优化三组全对率；两个目标不能混报。",
          "> B 会把阈值推高换取全对率，代价是 F1 下降。**必须先确认官方指标再选**。", "",
          "## 3. 逐标签阈值（calib 上搜索）", ""]
    for name, labels in GROUPS:
        pairs = ", ".join(f"{l}={t}" for l, t in zip(labels, prof_report[name]["thresholds"]))
        L.append(f"- **{name}**：{pairs}")
    L += ["", "## 4. 结论与后续", "",
          "- 情绪头是否达标（目标 Acc≥0.45、Macro-F1≥0.35）：见上表",
          "- 画像头是否达标（目标全对率≥0.40、interests F1≥0.45）：见上表",
          "- 下一步（T3）：把情绪/画像作为条件输入生成模型，并用分类头结果覆盖 LLM 的情绪输出",
          ""]
    report = "\n".join(L)
    with open(os.path.join(ROOT, "reports", "T2_报告.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n预测写入 {pred_path}")
    summary = {"split_seed": args.split_seed, "calib_n": len(idx_calib), "holdout_n": len(idx_test),
               "selection_reserved": selection_reserved, "emotion_holdout_acc": acc_t,
               "emotion_holdout_macro_f1": mf1_t, "emotion_full_acc": acc_f,
               "emotion_full_macro_f1": mf1_f, "profile_A_exact": exact_test,
               "profile_A_f1": {name: prof_report[name]["test_tuned"][2] for name, _ in GROUPS},
               "profile_B_exact": exact_b_test, "profile_B_f1": {name: f1_b[name][2] for name, _ in GROUPS},
               "empty_profile_exact": empty_test, "profile_B_threshold": float(gb_t)}
    with open(os.path.join(ROOT, "reports", "t2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
