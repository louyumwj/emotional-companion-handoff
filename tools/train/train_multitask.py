"""T2：多任务结构化预测头训练。

一个共享编码器 + 4 个头：
  emotion  16 分类（CrossEntropy，类别加权处理不均衡）
  traits   10 类多标签（BCE）
  interests 12 类多标签（BCE）
  style    11 类多标签（BCE）

用法（先冒烟再全量）：
  python tools/train/train_multitask.py --subset 200 --max-steps 20 --tag smoke
  python tools/train/train_multitask.py --epochs 2 --tag v1
"""
import argparse
import collections
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "train"))
from labels import EMOTIONS, INTERESTS, STYLES, TRAITS  # noqa: E402
from model_paths import portable_base, resolve_base  # noqa: E402
from validation_split import split_validation  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")


def build_text(history, max_chars=0):
    parts = []
    for t in history:
        prefix = "用户：" if t["role"] == "user" else "助手："
        parts.append(prefix + str(t["content"]).strip())
    text = "\n".join(parts)
    if max_chars and len(text) > max_chars:
        text = text[-max_chars:]
        cut = text.find("\n")
        if 0 <= cut < 60:
            text = text[cut + 1:]
    return text


class MultitaskDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=384):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length
        self.emo_idx = {e: i for i, e in enumerate(EMOTIONS)}
        self.groups = [("personality_traits", TRAITS), ("interests", INTERESTS), ("style", STYLES)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        text = build_text(r["history"], max_chars=0)  # 不预截断，交给 token 级左侧截断
        ids = self.tok(text, add_special_tokens=True)["input_ids"]
        # 关键：保留**最后** max_length 个 token（最新发言在末尾），而不是默认的前缀截断
        if len(ids) > self.max_length:
            ids = ids[:1] + ids[-(self.max_length - 1):]  # 保住 [CLS]
        attn = [1] * len(ids)
        pad = self.max_length - len(ids)
        ids = ids + [self.tok.pad_token_id] * pad
        attn = attn + [0] * pad
        prof = r["user_profile"]
        item = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "emotion": torch.tensor(self.emo_idx[r["emotion_label"]], dtype=torch.long),
        }
        for name, labels in self.groups:
            v = torch.zeros(len(labels))
            for x in prof.get(name, []):
                if x in labels:
                    v[labels.index(x)] = 1.0
            item[name] = v
        return item


class MultiTaskModel(nn.Module):
    def __init__(self, base, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head_emotion = nn.Linear(h, len(EMOTIONS))
        self.head_traits = nn.Linear(h, len(TRAITS))
        self.head_interests = nn.Linear(h, len(INTERESTS))
        self.head_style = nn.Linear(h, len(STYLES))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.drop(out.last_hidden_state[:, 0])
        return {
            "emotion": self.head_emotion(cls),
            "personality_traits": self.head_traits(cls),
            "interests": self.head_interests(cls),
            "style": self.head_style(cls),
        }


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    emo_tp = collections.Counter(); emo_fp = collections.Counter(); emo_fn = collections.Counter()
    group_tp = collections.Counter(); group_fp = collections.Counter(); group_fn = collections.Counter()
    exact_prof = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        pred = logits["emotion"].argmax(-1)
        gold = batch["emotion"]
        for g, p in zip(gold.tolist(), pred.tolist()):
            total += 1
            if g == p:
                correct += 1
                emo_tp[EMOTIONS[g]] += 1
            else:
                emo_fp[EMOTIONS[p]] += 1
                emo_fn[EMOTIONS[g]] += 1
        exact_batch = torch.ones(len(gold), dtype=torch.bool, device=gold.device)
        for name, labels in (("personality_traits", TRAITS), ("interests", INTERESTS),
                             ("style", STYLES)):
            prob = torch.sigmoid(logits[name])
            predm = (prob > 0.5).float()
            goldm = batch[name]
            tp = ((predm == 1) & (goldm == 1)).sum().item()
            fp = ((predm == 1) & (goldm == 0)).sum().item()
            fn = ((predm == 0) & (goldm == 1)).sum().item()
            group_tp[name] += tp; group_fp[name] += fp; group_fn[name] += fn
            exact_batch &= (predm == goldm).all(dim=1)
        exact_prof += exact_batch.sum().item()
    emo_acc = correct / max(1, total)
    f1s = [prf(emo_tp[e], emo_fp[e], emo_fn[e])[2] for e in EMOTIONS if
           emo_tp[e] + emo_fn[e] > 0]
    emo_macro = sum(f1s) / max(1, len(f1s))
    prof = {k: prf(group_tp[k], group_fp[k], group_fn[k]) for k in group_tp}
    return {"emotion_acc": emo_acc, "emotion_macro_f1": emo_macro,
            "profile_exact": exact_prof / max(1, total),
            "profile_f1": {k: v[2] for k, v in prof.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/chinese-roberta-wwm-ext")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 时用于冒烟测试")
    ap.add_argument("--subset", type=int, default=0, help=">0 时只用前 N 条训练")
    ap.add_argument("--lambda-profile", type=float, default=1.0)
    ap.add_argument("--out", default="models/t2_multitask")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--freeze-layers", type=int, default=0,
                    help="冻结编码器底部 N 层 + embedding，CPU 上可显著提速")
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"CPU 线程数：{torch.get_num_threads()}｜可用核：{os.cpu_count()}", flush=True)

    base = resolve_base(ROOT, args.base, override=args.base)
    train_rows = [json.loads(l) for l in open(os.path.join(PROC, "views", "train_cls.jsonl"),
                                              encoding="utf-8")]
    val_rows = [json.loads(l) for l in open(os.path.join(PROC, "views", "val_cls.jsonl"),
                                            encoding="utf-8")]
    idx_calib, idx_holdout = split_validation(val_rows, args.split_seed)
    val_rows = [val_rows[i] for i in idx_calib]
    if args.subset:
        train_rows = train_rows[:args.subset]
        val_rows = val_rows[:min(len(val_rows), max(100, args.subset // 2))]
    print(f"训练 {len(train_rows)} 条｜选检查点 {len(val_rows)} 条｜保留 holdout {len(idx_holdout)} 条", flush=True)

    tok = AutoTokenizer.from_pretrained(base)
    ds = MultitaskDataset(train_rows, tok, args.max_length)
    vds = MultitaskDataset(val_rows, tok, args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    vdl = DataLoader(vds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    model = MultiTaskModel(base)
    if args.freeze_layers > 0:
        for p in model.encoder.embeddings.parameters():
            p.requires_grad = False
        for layer in model.encoder.encoder.layer[:args.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        print(f"已冻结 embedding + 底部 {args.freeze_layers} 层", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 类别权重
    cnt = collections.Counter(r["emotion_label"] for r in train_rows)
    w = torch.tensor([len(train_rows) / (len(EMOTIONS) * max(1, cnt.get(e, 0)))
                      for e in EMOTIONS], dtype=torch.float)
    w = (w / w.mean()).clamp(0.2, 8.0).to(device)
    ce = nn.CrossEntropyLoss(weight=w)
    bce = nn.BCEWithLogitsLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total_steps = max(1, int(len(dl) * args.epochs))
    if args.max_steps:
        total_steps = args.max_steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps,
                                                pct_start=0.06)
    print(f"总步数：{total_steps}｜batch={args.batch_size}｜max_len={args.max_length}", flush=True)

    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    step = 0
    t0 = time.time()
    history = []
    best = -1.0
    stop = False
    for ep in range(math.ceil(args.epochs) if not args.max_steps else 10 ** 6):
        model.train()
        run_loss = 0.0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = ce(logits["emotion"], batch["emotion"])
            for name in ("personality_traits", "interests", "style"):
                loss = loss + args.lambda_profile * bce(logits[name], batch[name])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run_loss += loss.item()
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"  ep{ep+1} step {step}/{total_steps} loss={run_loss/50:.4f} "
                      f"({el:.0f}s, {step/el:.2f} step/s)", flush=True)
                run_loss = 0.0
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        m = evaluate(model, vdl, device)
        m["epoch"] = ep + 1
        m["step"] = step
        history.append(m)
        print(f"[验证] ep{ep+1} 情绪Acc={m['emotion_acc']:.4f} "
              f"MacroF1={m['emotion_macro_f1']:.4f} 画像全对={m['profile_exact']:.4f} "
              f"F1={ {k: round(v,3) for k,v in m['profile_f1'].items()} }", flush=True)
        score = m["emotion_acc"] + m["emotion_macro_f1"] + m["profile_exact"]
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "base": portable_base(ROOT, base),
                        "max_length": args.max_length, "selection_split": "conversation_calib",
                        "split_seed": args.split_seed}, os.path.join(ROOT, args.out, "best.pt"))
            print("  ↑ 保存最优检查点", flush=True)
        if stop:
            break

    with open(os.path.join(ROOT, args.out, f"train_log_{args.tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump({"args": vars(args), "history": history,
                   "elapsed_sec": round(time.time() - t0, 1)}, f, ensure_ascii=False, indent=2)
    print(f"\n完成，用时 {time.time()-t0:.0f}s，模型保存至 {args.out}/best.pt")


if __name__ == "__main__":
    main()
