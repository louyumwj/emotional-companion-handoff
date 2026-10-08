"""memory_refs 结构分析：判断记忆 ID 是「会话内编号」还是「全局编号」。

这一步决定 memory_refs 这个字段能不能从公开输入里学到。
"""
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(ROOT, "data", "incoming", "训练-验证-数据集", "train", "train_public.jsonl")


def main():
    conv_blocks = {}          # conv_id -> sorted list of int ids used
    id_to_convs = collections.defaultdict(set)
    anomalies = []
    refs_per_point_pairs = []
    n = 0
    with open(PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            n += 1
            ids = []
            for p in c["prediction_points"]:
                gap = p["assistant_response_turn_id"] - p["target_user_turn_id"]
                if gap != 1:
                    anomalies.append((c["conversation_id"], p["target_user_turn_id"],
                                      p["assistant_response_turn_id"]))
                refs = [int(m.split("_")[1]) for m in p["target"]["memory_refs"]]
                ids.extend(refs)
                refs_per_point_pairs.append((c["conversation_id"], len(refs), refs))
                # 该预测点之前有几轮 user（可作为"可用记忆"上界）
            if ids:
                conv_blocks[c["conversation_id"]] = sorted(set(ids))
                for i in set(ids):
                    id_to_convs[i].add(c["conversation_id"])

    print(f"扫描会话数：{n}")
    print(f"有记忆引用的会话数：{len(conv_blocks)}")
    print(f"出现过的不同记忆 ID 数：{len(id_to_convs)}")

    # 1) 每个会话的记忆 ID 是否连续
    contiguous = 0
    span_vs_count = []
    for cid, ids in conv_blocks.items():
        span = ids[-1] - ids[0] + 1
        span_vs_count.append((span, len(ids)))
        if span == len(ids):
            contiguous += 1
    print(f"\n[1] 记忆ID在会话内完全连续的会话数：{contiguous}/{len(conv_blocks)}")
    ratios = [c / s for s, c in span_vs_count if s]
    if ratios:
        ratios.sort()
        print(f"    密度(实际个数/跨度) 中位数={ratios[len(ratios)//2]:.3f} "
              f"min={ratios[0]:.3f} max={ratios[-1]:.3f}")

    # 2) 同一 ID 被多少会话共用
    shared = sum(1 for v in id_to_convs.values() if len(v) > 1)
    print(f"\n[2] 被多个会话共用的记忆 ID 数：{shared} / {len(id_to_convs)}"
          f"（{shared/max(1,len(id_to_convs)):.1%}）")
    multi = sorted(((len(v), k) for k, v in id_to_convs.items()), reverse=True)[:8]
    print(f"    跨会话使用最多的 ID：{[(f'mem_{k:06d}', c) for c, k in multi]}")

    # 3) 全局 ID 是否从小到大单调分配（全局计数器假说）
    all_ids = sorted(id_to_convs)
    print(f"\n[3] 全局 ID 范围：mem_{all_ids[0]:06d} ~ mem_{all_ids[-1]:06d}"
          f"（共 {len(all_ids)} 个不同值，跨度 {all_ids[-1]-all_ids[0]+1}）")
    print(f"    缺失率 = {1 - len(all_ids)/(all_ids[-1]-all_ids[0]+1):.1%}")

    # 4) 会话首个 ID 与会话序号的关系（前 10 个会话）
    print("\n[4] 前 10 个会话使用的记忆 ID 区间：")
    for i, (cid, ids) in enumerate(list(conv_blocks.items())[:10]):
        print(f"    {cid}  {len(ids):>3} 个  mem_{ids[0]:06d} ~ mem_{ids[-1]:06d}")

    # 5) 异常预测点
    print(f"\n[5] 非相邻的预测点（gap != 1）：{len(anomalies)}")
    for a in anomalies[:10]:
        print("   ", a)

    # 6) 同一会话内同一 ID 是否被重复引用
    dup_in_conv = 0
    for cid, ids in conv_blocks.items():
        pass
    seq = collections.defaultdict(list)
    for cid, k, refs in refs_per_point_pairs:
        seq[cid].append(refs)
    for cid, lst in seq.items():
        flat = [x for r in lst for x in r]
        if len(flat) != len(set(flat)):
            dup_in_conv += 1
    print(f"\n[6] 同一会话内同一记忆 ID 被多个预测点重复引用的会话数：{dup_in_conv}")

    # 7) 首个预测点 vs 引用的最大 ID（看是否只引用"过去"）
    print("\n[7] 抽样：预测点位置 与 引用ID 的关系")
    c = 0
    for cid, k, refs in refs_per_point_pairs:
        if k >= 2 and c < 8:
            print(f"    {cid}  引用 {k} 个: {[f'mem_{r:06d}' for r in refs]}")
            c += 1


if __name__ == "__main__":
    sys.exit(main())
