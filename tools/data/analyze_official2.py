"""补充分析：val 记忆编号范围、预测点位置规律、画像是否随轮次累积、未来上文可得性。"""
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "data", "incoming", "训练-验证-数据集")


def load(split):
    p = os.path.join(BASE, split, f"{split}_public.jsonl")
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def mem_range(split):
    lo, hi, n = 10 ** 9, -1, 0
    blocks = []
    for c in load(split):
        ids = []
        for p in c["prediction_points"]:
            for m in p["target"]["memory_refs"]:
                ids.append(int(m.split("_")[1]))
        if ids:
            lo = min(lo, min(ids)); hi = max(hi, max(ids)); n += 1
            blocks.append((c["conversation_id"], min(ids), max(ids), len(set(ids))))
    print(f"【{split}】记忆ID范围 mem_{lo:06d} ~ mem_{hi:06d}，有引用的会话 {n}")
    print(f"     前 5 个块: {blocks[:5]}")
    print(f"     后 5 个块: {blocks[-5:]}")
    return lo, hi


print("=" * 70)
tlo, thi = mem_range("train")
vlo, vhi = mem_range("val")
print(f"\n训练与验证的记忆ID范围是否重叠: {not (vhi < tlo or vlo > thi)}")

print("\n" + "=" * 70)
odd = even = 0
has_future = 0
last_turn = 0
prof_growth = collections.Counter()
first_turn_is_point = 0
total_points = 0
resp_eq_profile = 0
for c in load("train"):
    turns = c["turns"]
    by_id = {t["turn_id"]: t for t in turns}
    max_turn = max(by_id)
    prev_prof = None
    for p in c["prediction_points"]:
        total_points += 1
        uid = p["target_user_turn_id"]
        aid = p["assistant_response_turn_id"]
        if uid % 2 == 1:
            odd += 1
        else:
            even += 1
        if aid >= max_turn:
            last_turn += 1
        else:
            has_future += 1
        if uid == 1:
            first_turn_is_point += 1
        prof = p["target"]["user_profile"]
        size = (len(prof["personality_traits"]), len(prof["interests"]), len(prof["style"]))
        if prev_prof is not None:
            grew = sum(size) - sum(prev_prof)
            prof_growth[grew] += 1
        prev_prof = size

print(f"target_user_turn_id 为奇数: {odd}   偶数: {even}   共 {total_points}")
print(f"预测点的回复是会话最后一轮: {last_turn}   之后还有内容: {has_future}")
print(f"预测点落在第 1 轮（完全无历史）: {first_turn_is_point}")
print(f"同一会话内相邻预测点的画像规模变化: {dict(sorted(prof_growth.items()))}")
