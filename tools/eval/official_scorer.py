"""官方格式的本地评测器（四个字段分别计分）。

支持两种预测格式：
  {"instance_id": "...", "response_text": ..., "emotion_label": ..., "user_profile": {...}, "memory_refs": [...]}
  {"instance_id": "...", "target": {"response_text": ..., ...}}   ← 官方样式

用法：
  python tools/eval/official_scorer.py --pred preds.jsonl --gold data/processed/official_val.jsonl \
      --out reports/score.md
"""
import argparse
import collections
import json
import math
import os
import sys

EMOTIONS = ["joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise", "mixed",
            "sadness", "loneliness", "anxiety", "anger", "fear", "disgust", "shame",
            "helplessness"]
PROFILE_KEYS = ["personality_traits", "interests", "style"]


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def normalize_pred(obj):
    t = obj.get("target", obj)
    prof = t.get("user_profile") or {}
    return {
        "instance_id": obj.get("instance_id") or obj.get("conversation_id"),
        "response_text": t.get("response_text", "") or "",
        "emotion_label": t.get("emotion_label", ""),
        "user_profile": {k: list(prof.get(k) or []) for k in PROFILE_KEYS},
        "memory_refs": list(t.get("memory_refs") or []),
    }


def gold_of(obj):
    t = obj["target"]
    return {
        "response_text": t["response_text"],
        "emotion_label": t["emotion_label"],
        "user_profile": t["user_profile"],
        "memory_refs": t["memory_refs"],
    }


# ---------------- 文本指标（字符级，中文适用） ----------------
def ngrams(text, n):
    text = "".join(text.split())
    return [text[i:i + n] for i in range(max(0, len(text) - n + 1))]


def rouge_n(pred, gold, n):
    p, g = collections.Counter(ngrams(pred, n)), collections.Counter(ngrams(gold, n))
    if not g:
        return 1.0 if not p else 0.0
    overlap = sum((p & g).values())
    return overlap / sum(g.values())


def lcs_len(a, b):
    a, b = "".join(a.split()), "".join(b.split())
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            cur[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(b)]


def rouge_l(pred, gold):
    denom = len("".join(gold.split()))
    return lcs_len(pred, gold) / denom if denom else 0.0


def bleu(pred, gold, max_n=4):
    pred, gold = "".join(pred.split()), "".join(gold.split())
    if not pred or not gold:
        return 0.0
    score = 0.0
    for n in range(1, max_n + 1):
        p = collections.Counter(ngrams(pred, n))
        g = collections.Counter(ngrams(gold, n))
        total = sum(p.values())
        if total == 0:
            return 0.0
        score += math.log((sum((p & g).values()) + 1) / (total + 1)) / max_n
    bp = min(0.0, 1 - len(gold) / len(pred))
    return math.exp(score + bp)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def align_predictions(gold_rows, pred_rows, limit=0, allow_partial=False):
    gold_ids = [r["instance_id"] for r in gold_rows]
    if len(set(gold_ids)) != len(gold_ids):
        raise ValueError("Duplicate instance_id in gold file")
    if limit < 0:
        raise ValueError("--limit must be nonnegative")
    normalized = [normalize_pred(r) for r in pred_rows]
    ids = [r["instance_id"] for r in normalized]
    if any(not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Missing or duplicate instance_id in predictions")
    unknown = set(ids) - set(gold_ids)
    if unknown:
        raise ValueError(f"Unknown prediction IDs: {sorted(unknown)[:3]}")
    expected = gold_ids[:limit] if limit else gold_ids
    pred_by_id = {r["instance_id"]: r for r in normalized}
    missing = [i for i in expected if i not in pred_by_id]
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)}/{len(expected)} predictions. Use --allow-partial only for smoke checks.")
    selected = [pred_by_id[i] for i in expected if i in pred_by_id]
    if not selected:
        raise ValueError("No predictions to score")
    return selected, len(expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-partial", action="store_true", help="Explicit smoke-test subset; report coverage")
    args = ap.parse_args()

    gold_rows = load_jsonl(args.gold)
    golds = {g["instance_id"]: gold_of(g) for g in gold_rows}
    preds, expected_n = align_predictions(gold_rows, load_jsonl(args.pred), args.limit, args.allow_partial)

    n = len(preds)
    ids = [p["instance_id"] for p in preds]

    # ---- 情绪 ----
    emo_tp = collections.Counter()
    emo_fp = collections.Counter()
    emo_fn = collections.Counter()
    emo_correct = 0
    confusion = collections.Counter()
    for p in preds:
        g = golds[p["instance_id"]]["emotion_label"]
        q = p["emotion_label"]
        if q == g:
            emo_correct += 1
            emo_tp[g] += 1
        else:
            emo_fp[q] += 1
            emo_fn[g] += 1
            confusion[(g, q)] += 1
    emo_acc = emo_correct / n if n else 0.0
    per_class = {}
    for e in EMOTIONS:
        if emo_tp[e] or emo_fp[e] or emo_fn[e]:
            per_class[e] = prf(emo_tp[e], emo_fp[e], emo_fn[e])
    macro_f1 = sum(v[2] for v in per_class.values()) / len(per_class) if per_class else 0.0

    # ---- 画像（多标签） ----
    prof_stats = {}
    exact_profile = 0
    for k in PROFILE_KEYS:
        tp = fp = fn = 0
        for p in preds:
            gset = set(golds[p["instance_id"]]["user_profile"][k])
            qset = set(p["user_profile"][k])
            tp += len(gset & qset)
            fp += len(qset - gset)
            fn += len(gset - qset)
        prof_stats[k] = prf(tp, fp, fn)
    for p in preds:
        g = golds[p["instance_id"]]["user_profile"]
        if all(set(g[k]) == set(p["user_profile"][k]) for k in PROFILE_KEYS):
            exact_profile += 1
    profile_exact = exact_profile / n if n else 0.0

    # ---- 记忆引用 ----
    m_tp = m_fp = m_fn = 0
    mem_exact = 0
    cnt_abs_err = 0
    for p in preds:
        gset = set(golds[p["instance_id"]]["memory_refs"])
        qset = set(p["memory_refs"])
        m_tp += len(gset & qset)
        m_fp += len(qset - gset)
        m_fn += len(gset - qset)
        if gset == qset:
            mem_exact += 1
        cnt_abs_err += abs(len(gset) - len(qset))
    mem_p, mem_r, mem_f = prf(m_tp, m_fp, m_fn)
    mem_exact_rate = mem_exact / n if n else 0.0
    mem_count_mae = cnt_abs_err / n if n else 0.0

    # ---- 回复文本 ----
    r1 = r2 = rl = bl = 0.0
    len_pred = len_gold = 0
    for p in preds:
        g = golds[p["instance_id"]]["response_text"]
        q = p["response_text"]
        r1 += rouge_n(q, g, 1)
        r2 += rouge_n(q, g, 2)
        rl += rouge_l(q, g)
        bl += bleu(q, g, 4)
        len_pred += len(q)
        len_gold += len(g)
    if n:
        r1, r2, rl, bl = r1 / n, r2 / n, rl / n, bl / n
        len_pred, len_gold = len_pred / n, len_gold / n

    lines = [
        "# 官方数据集本地评测报告", "",
        f"- 预测文件：`{os.path.basename(args.pred)}`",
        f"- 参考文件：`{os.path.basename(args.gold)}`",
        f"- 实例数：**{n}**", "",
        f"- 覆盖率：**{n}/{expected_n} ({n / expected_n:.2%})**；模式："
        + ("显式子集冒烟" if args.allow_partial else "完整覆盖校验"), "",
        "## 1. 情绪标签（16 分类）", "",
        f"- **Accuracy：{emo_acc:.4f}**",
        f"- **Macro-F1：{macro_f1:.4f}**", "",
        "| 情绪 | P | R | F1 | 支持数 |", "|---|---|---|---|---|",
    ]
    for e, (p_, r_, f_) in sorted(per_class.items(), key=lambda x: -x[1][1]):
        sup = emo_tp[e] + emo_fn[e]
        lines.append(f"| {e} | {p_:.3f} | {r_:.3f} | {f_:.3f} | {sup} |")

    lines += ["", "## 2. 用户画像（多标签）", "",
              "| 字段 | P | R | F1 |", "|---|---|---|---|"]
    for k, (p_, r_, f_) in prof_stats.items():
        lines.append(f"| {k} | {p_:.3f} | {r_:.3f} | {f_:.3f} |")
    lines += [f"| **三数组完全一致率** | | | **{profile_exact:.4f}** |", ""]

    lines += ["## 3. 记忆引用（集合匹配）", "",
              f"- P = {mem_p:.4f}　R = {mem_r:.4f}　**F1 = {mem_f:.4f}**",
              f"- 完全一致率：{mem_exact_rate:.4f}",
              f"- 引用个数平均绝对误差：{mem_count_mae:.3f}", ""]

    lines += ["## 4. 回复文本（与参考回复比对）", "",
              f"- ROUGE-1 = {r1:.4f}　ROUGE-2 = {r2:.4f}　ROUGE-L = {rl:.4f}",
              f"- char-BLEU-4 = {bl:.4f}",
              f"- 平均长度：预测 {len_pred:.1f} 字 vs 参考 {len_gold:.1f} 字", "",
              "> 说明：文本相似度只是粗代理，真实评分很可能是 LLM-as-judge，"
              "需要用 T2 的评审脚本补充。", ""]

    if confusion:
        lines += ["## 5. 主要混淆（真实 → 预测，Top 12）", "",
                  "| 真实 | 预测 | 次数 |", "|---|---|---|"]
        for (g, q), c in confusion.most_common(12):
            lines.append(f"| {g} | {q} | {c} |")

    report = "\n".join(lines)
    print(report)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n报告已写入 {args.out}")

    summary = {"n": n, "expected_n": expected_n, "coverage": n / expected_n,
               "emotion_acc": emo_acc, "emotion_macro_f1": macro_f1,
               "profile_exact": profile_exact, "memory_f1": mem_f,
               "memory_exact": mem_exact_rate, "rouge_l": rl, "bleu4": bl}
    with open(os.path.join(os.path.dirname(args.out) or ".", "_last_score.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
