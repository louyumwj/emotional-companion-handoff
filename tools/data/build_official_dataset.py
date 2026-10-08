"""把官方 train/val 转成规范化的「实例」格式。

关键点：官方公开文件里 turns[] **已经包含**待预测的助手回复（18657/18657 条完全一致），
说明推理时该轮会被抹掉。因此本脚本统一按「截断到 target_user_turn_id」构造 history，
训练与推理形状一致，避免答案泄漏。

产出：
  data/processed/official_{split}.jsonl        规范化实例（含全部标签）
  data/processed/views/{split}_resp_sft.jsonl  回复生成视图（Qwen chat 格式）
  data/processed/views/{split}_cls.jsonl       三分类头视图（情绪/画像/记忆）
  data/processed/build_stats.json              构建统计
"""
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "data", "incoming", "训练-验证-数据集")
OUT = os.path.join(ROOT, "data", "processed")
VIEWS = os.path.join(OUT, "views")

SYSTEM_PROMPT = (
    "你是面向中国大学生的情感陪伴伙伴。请先准确接住对方的情绪，再决定是否提供建议；"
    "语气自然、口语化，一次回复 2~4 句；不做诊断、不开药、不下病理结论。"
)


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_split(split, stats):
    convs = list(iter_jsonl(os.path.join(BASE, split, f"{split}_public.jsonl")))
    instances, resp_sft, cls_rows = [], [], []
    profile_consistent = same_cnt = multi_pt_conv = 0
    skipped_anomaly = 0

    for c in convs:
        turns = c["turns"]
        by_id = {t["turn_id"]: t for t in turns}
        profiles = []
        for idx, p in enumerate(c["prediction_points"], start=1):
            uid = p["target_user_turn_id"]
            aid = p["assistant_response_turn_id"]
            if aid - uid != 1 or uid not in by_id or aid not in by_id:
                skipped_anomaly += 1
                continue
            # 截断到目标用户轮（含），杜绝未来信息与答案泄漏
            history = [{"role": t["role"], "content": t["content"]}
                       for t in turns if t["turn_id"] <= uid]
            if not history or history[-1]["role"] != "user":
                skipped_anomaly += 1
                continue

            tgt = p["target"]
            inst_id = f"{c['conversation_id']}#p{idx}"
            inst = {
                "instance_id": inst_id,
                "conversation_id": c["conversation_id"],
                "split": split,
                "history": history,
                "n_history_turns": len(history),
                "target_user_turn_id": uid,
                "assistant_response_turn_id": aid,
                "target": {
                    "response_text": tgt["response_text"],
                    "emotion_label": tgt["emotion_label"],
                    "user_profile": tgt["user_profile"],
                    "memory_refs": tgt["memory_refs"],
                },
            }
            instances.append(inst)
            profiles.append(tgt["user_profile"])

            resp_sft.append({
                "instance_id": inst_id,
                "messages": ([{"role": "system", "content": SYSTEM_PROMPT}]
                             + history
                             + [{"role": "assistant", "content": tgt["response_text"]}]),
            })
            cls_rows.append({
                "instance_id": inst_id,
                "history": history,
                "emotion_label": tgt["emotion_label"],
                "user_profile": tgt["user_profile"],
                "memory_refs": tgt["memory_refs"],
                "n_memory_refs": len(tgt["memory_refs"]),
            })

        if len(profiles) > 1:
            multi_pt_conv += 1
            if all(pp == profiles[0] for pp in profiles):
                profile_consistent += 1

    stats[split] = {
        "instances": len(instances),
        "skipped_anomaly": skipped_anomaly,
        "conversations_with_multiple_points": multi_pt_conv,
        "conversations_with_identical_profile_across_points": profile_consistent,
        "profile_consistency_rate": round(profile_consistent / max(1, multi_pt_conv), 4),
    }

    os.makedirs(VIEWS, exist_ok=True)
    with open(os.path.join(OUT, f"official_{split}.jsonl"), "w", encoding="utf-8") as f:
        for r in instances:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(VIEWS, f"{split}_resp_sft.jsonl"), "w", encoding="utf-8") as f:
        for r in resp_sft:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(VIEWS, f"{split}_cls.jsonl"), "w", encoding="utf-8") as f:
        for r in cls_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return instances


def main():
    os.makedirs(OUT, exist_ok=True)
    stats = {}
    for split in ("train", "val"):
        inst = build_split(split, stats)
        print(f"[{split}] 实例 {len(inst)} 条 -> data/processed/official_{split}.jsonl")

    # 标签分布（以实例为单位）
    import collections as C
    tr = [json.loads(l) for l in open(os.path.join(OUT, "official_train.jsonl"), encoding="utf-8")]
    emo = C.Counter(i["target"]["emotion_label"] for i in tr)
    stats["train_emotion_dist"] = dict(emo.most_common())
    stats["train_emotion_majority"] = emo.most_common(1)[0]
    stats["train_emotion_majority_acc"] = round(emo.most_common(1)[0][1] / len(tr), 4)

    n = len(tr)
    stats["train_profile_empty_rate"] = {
        k: round(sum(1 for i in tr if not i["target"]["user_profile"][k]) / n, 4)
        for k in ("personality_traits", "interests", "style")
    }
    stats["train_memory_empty_rate"] = round(
        sum(1 for i in tr if not i["target"]["memory_refs"]) / n, 4)
    stats["train_history_turns_dist"] = dict(
        sorted(C.Counter(i["n_history_turns"] for i in tr).items())[:12])

    with open(os.path.join(OUT, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n===== 构建统计 =====")
    for split in ("train", "val"):
        print(f"  [{split}] {stats[split]}")
    print(f"\n  情绪多数类: {stats['train_emotion_majority']} "
          f"→ 全猜它可得 {stats['train_emotion_majority_acc']:.1%}")
    print(f"  画像为空的比例: {stats['train_profile_empty_rate']}")
    print(f"  记忆引用为空的比例: {stats['train_memory_empty_rate']}")
    print(f"\n统计写入 data/processed/build_stats.json")


if __name__ == "__main__":
    sys.exit(main())
