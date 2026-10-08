"""官方数据集结构与统计特征分析。

输出：
  - 控制台统计
  - data/incoming/_analysis.json         机器可读的统计
  - data/incoming/_sample_records.txt    2 条完整样本（截断展示）
"""
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFICIAL = os.path.join(ROOT, "data", "incoming", "训练-验证-数据集")


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def analyze(split):
    path = os.path.join(OFFICIAL, split, f"{split}_public.jsonl")
    convs = list(iter_jsonl(path))
    st = {
        "split": split, "conversations": len(convs),
        "turns_total": 0, "points_total": 0,
        "turns_per_conv": [], "points_per_conv": [],
        "resp_len_user": [], "resp_len_assistant": [], "resp_len_target": [],
        "emotion": collections.Counter(), "traits": collections.Counter(),
        "interests": collections.Counter(), "styles": collections.Counter(),
        "n_traits": collections.Counter(), "n_interests": collections.Counter(),
        "n_styles": collections.Counter(), "n_memrefs": collections.Counter(),
        "mem_ids": collections.Counter(),
        "resp_turn_gap": collections.Counter(),
        "target_equals_turn": 0, "target_missing_in_turns": 0, "target_not_adjacent": 0,
        "first_target_turn": collections.Counter(),
        "user_roles": 0, "assistant_roles": 0, "bad_roles": 0,
        "memid_by_conv": {}, "conv_with_mem": 0,
    }
    for c in convs:
        turns = c["turns"]
        pts = c["prediction_points"]
        st["turns_total"] += len(turns)
        st["points_total"] += len(pts)
        st["turns_per_conv"].append(len(turns))
        st["points_per_conv"].append(len(pts))
        by_id = {}
        for t in turns:
            by_id[t["turn_id"]] = t
            if t["role"] == "user":
                st["user_roles"] += 1
                st["resp_len_user"].append(len(t["content"]))
            elif t["role"] == "assistant":
                st["assistant_roles"] += 1
                st["resp_len_assistant"].append(len(t["content"]))
            else:
                st["bad_roles"] += 1
        if not pts:
            continue
        st["first_target_turn"].update([pts[0]["target_user_turn_id"]])
        conv_mem = set()
        for p in pts:
            gid = p["assistant_response_turn_id"] - p["target_user_turn_id"]
            st["resp_turn_gap"].update([gid])
            if gid != 1:
                st["target_not_adjacent"] += 1
            tgt = p["target"]
            st["resp_len_target"].append(len(tgt["response_text"]))
            rt = by_id.get(p["assistant_response_turn_id"])
            if rt is None:
                st["target_missing_in_turns"] += 1
            elif rt["content"].strip() == tgt["response_text"].strip():
                st["target_equals_turn"] += 1
            st["emotion"].update([tgt["emotion_label"]])
            prof = tgt["user_profile"]
            st["n_traits"].update([len(prof["personality_traits"])])
            st["n_interests"].update([len(prof["interests"])])
            st["n_styles"].update([len(prof["style"])])
            st["traits"].update(prof["personality_traits"])
            st["interests"].update(prof["interests"])
            st["styles"].update(prof["style"])
            mrefs = tgt["memory_refs"]
            st["n_memrefs"].update([len(mrefs)])
            st["mem_ids"].update(mrefs)
            conv_mem.update(mrefs)
        if conv_mem:
            st["conv_with_mem"] += 1
            st["memid_by_conv"][c["conversation_id"]] = len(conv_mem)

    def dist(xs):
        if not xs:
            return {}
        xs = sorted(xs)
        n = len(xs)
        return {"min": xs[0], "p25": xs[n // 4], "median": xs[n // 2],
                "p75": xs[3 * n // 4], "max": xs[-1], "mean": round(sum(xs) / n, 2)}

    out = {
        "split": split,
        "conversations": st["conversations"],
        "turns_total": st["turns_total"],
        "prediction_points": st["points_total"],
        "turns_per_conv": dist(st["turns_per_conv"]),
        "points_per_conv": dist(st["points_per_conv"]),
        "user_turn_chars": dist(st["resp_len_user"]),
        "assistant_turn_chars": dist(st["resp_len_assistant"]),
        "target_response_chars": dist(st["resp_len_target"]),
        "emotion_label_dist": dict(st["emotion"].most_common()),
        "profile_array_sizes": {"traits": dict(st["n_traits"]), "interests": dict(st["n_interests"]),
                                "style": dict(st["n_styles"])},
        "traits_dist": dict(st["traits"].most_common()),
        "interests_dist": dict(st["interests"].most_common()),
        "style_dist": dict(st["styles"].most_common()),
        "memory_refs_per_point": dict(sorted(st["n_memrefs"].items())),
        "distinct_mem_ids": len(st["mem_ids"]),
        "mem_id_top": st["mem_ids"].most_common(10),
        "conversations_with_memory": st["conv_with_mem"],
        "resp_turn_gap_dist": dict(st["resp_turn_gap"]),
        "target_text_equals_turn_text": st["target_equals_turn"],
        "target_turn_missing": st["target_missing_in_turns"],
        "not_adjacent": st["target_not_adjacent"],
        "first_prediction_target_turn_dist": dict(sorted(st["first_target_turn"].items())),
        "roles": {"user": st["user_roles"], "assistant": st["assistant_roles"],
                  "bad": st["bad_roles"]},
    }
    return convs, out


def dump_samples(convs, path, n=2):
    with open(path, "w", encoding="utf-8") as f:
        for c in convs[:n]:
            f.write("=" * 80 + "\n")
            f.write(f"conversation_id: {c['conversation_id']}\n")
            f.write(f"turns: {len(c['turns'])}   prediction_points: {len(c['prediction_points'])}\n")
            f.write("-" * 80 + "\n")
            for t in c["turns"]:
                f.write(f"[{t['turn_id']:>3}] {t['role']:<9} | {t['content']}\n")
            f.write("-" * 80 + "\n预测目标\n")
            for p in c["prediction_points"]:
                f.write(json.dumps(p, ensure_ascii=False, indent=2) + "\n")


def main():
    all_stats = {}
    for split in ("train", "val"):
        convs, st = analyze(split)
        all_stats[split] = st
        print("=" * 78)
        print(f"【{split}】会话 {st['conversations']} 条 | 总轮次 {st['turns_total']} | "
              f"预测点 {st['prediction_points']}")
        print(f"  每会话轮次数: {st['turns_per_conv']}")
        print(f"  每会话预测点数: {st['points_per_conv']}")
        print(f"  用户发言长度: {st['user_turn_chars']}")
        print(f"  助手发言长度: {st['assistant_turn_chars']}")
        print(f"  目标回复长度: {st['target_response_chars']}")
        print(f"  角色统计: {st['roles']}")
        print(f"  目标回复轮 = 用户轮+1 的分布: {st['resp_turn_gap_dist']}")
        print(f"  目标文本与 turns 中同轮文本完全一致: {st['target_text_equals_turn_text']} "
              f"（缺轮: {st['target_turn_missing']}）")
        print(f"  首个预测点位置分布: {st['first_prediction_target_turn_dist']}")
        print(f"\n  情绪标签分布（{len(st['emotion_label_dist'])} 类）:")
        for k, v in st["emotion_label_dist"].items():
            print(f"     {k:<16} {v:>6}  {v/max(1,st['prediction_points']):.1%}")
        print(f"\n  画像数组长度: {st['profile_array_sizes']}")
        print(f"  性格特质分布: {st['traits_dist']}")
        print(f"  兴趣分布: {st['interests_dist']}")
        print(f"  表达风格分布: {st['style_dist']}")
        print(f"\n  每个预测点的 memory_refs 个数: {st['memory_refs_per_point']}")
        print(f"  出现过的不同 memory id 数: {st['distinct_mem_ids']}")
        print(f"  最常见 memory id: {st['mem_id_top']}")
        print(f"  有记忆引用的会话数: {st['conversations_with_memory']} / {st['conversations']}")
        if split == "train":
            dump_samples(convs, os.path.join(ROOT, "data", "incoming", "_sample_records.txt"))
    with open(os.path.join(ROOT, "data", "incoming", "_analysis.json"), "w",
              encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print("\n统计已写入 data/incoming/_analysis.json，样本已写入 data/incoming/_sample_records.txt")


if __name__ == "__main__":
    sys.exit(main())
