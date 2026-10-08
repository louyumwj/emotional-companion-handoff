"""T3：Qwen2.5 LoRA SFT（官方回复语料）。

本机无 GPU，此脚本用同一份代码在 CPU 上跑小模型冒烟验证；
换到 GPU（赛事 20 卡时）只需把 --model 指向 Qwen2.5-7B-Instruct 并放大 batch。

用法：
  # CPU 冒烟（0.5B + 300 条 + 60 步）
  python tools/train/train_sft.py --model models/Qwen2.5-0.5B-Instruct \
      --subset 300 --max-steps 60 --tag smoke
  # GPU 全量（7B）
  python tools/train/train_sft.py --model models/Qwen2.5-7B-Instruct \
      --epochs 2 --batch-size 2 --grad-accum 8 --max-len 1536 --tag v1
"""
import argparse
import json
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "train"))
from model_paths import portable_base  # noqa: E402
SFT_DIR = os.path.join(ROOT, "data", "processed", "views")


class SFTDataset(Dataset):
    def __init__(self, rows, tok, max_len=1024):
        self.rows = rows
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        msgs = self.rows[i]["messages"]
        assert msgs[-1]["role"] == "assistant", "最后一条必须是 assistant"
        prompt_msgs, target = msgs[:-1], msgs[-1]["content"]
        prompt = self.tok.apply_chat_template(prompt_msgs, tokenize=False,
                                              add_generation_prompt=True)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = self.tok(target, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]

        # 关键：截断必须发生在 prompt 一侧（从左侧丢最早的上下文），
        # 绝不能截掉目标回复，否则 labels 全为 -100，交叉熵会产生 NaN。
        if len(t_ids) >= self.max_len - 16:          # 目标本身就超长：截目标尾部
            t_ids = t_ids[: self.max_len - 16]
        budget = self.max_len - len(t_ids)
        if len(p_ids) > budget:
            p_ids = p_ids[-budget:]                   # 保留最近的对话上下文
        ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + t_ids
        assert any(l != -100 for l in labels), "labels 全被屏蔽，样本不可用"
        return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default=os.path.join(SFT_DIR, "train_resp_sft.jsonl"))
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--out", default="models/t3_lora")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    model_path = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"设备={device} dtype={dtype} 线程={torch.get_num_threads()}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype,
                                                 trust_remote_code=True)
    model.config.use_cache = False

    from peft import LoraConfig, get_peft_model
    try:
        lc = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout, bias="none",
                        task_type="CAUSAL_LM", target_modules="all-linear")
        model = get_peft_model(model, lc)
    except Exception as exc:
        print(f"all-linear 失败({exc})，改用显式模块名", flush=True)
        lc = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout, bias="none",
                        task_type="CAUSAL_LM",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lc)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数 {trainable/1e6:.2f}M / 总参数 {total/1e6:.1f}M "
          f"({trainable/total:.3%})", flush=True)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.subset:
        rows = rows[:args.subset]
    print(f"训练样本 {len(rows)} 条", flush=True)
    ds = SFTDataset(rows, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=Collator(tok.pad_token_id), num_workers=0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = max(1, math.ceil(len(dl) / args.grad_accum))
    total_steps = args.max_steps or max(1, int(steps_per_epoch * args.epochs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps,
                                                       eta_min=args.lr * 0.1)
    print(f"优化步数 {total_steps}（batch {args.batch_size} × accum {args.grad_accum}）",
          flush=True)

    model.to(device)
    model.train()
    t0, step, micro, run = time.time(), 0, 0, 0.0
    opt.zero_grad(set_to_none=True)
    done = False
    for ep in range(max(1, math.ceil(args.epochs)) if not args.max_steps else 10 ** 6):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            run += out.loss.item()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == 1:
                    el = time.time() - t0
                    print(f"  ep{ep+1} step {step}/{total_steps} "
                          f"loss={run/ (10*args.grad_accum if step>1 else args.grad_accum):.4f} "
                          f"({el:.0f}s, {step/el*60:.1f} 步/分)", flush=True)
                    run = 0.0
                if step >= total_steps:
                    done = True
                    break
        if done:
            break

    outdir = os.path.join(ROOT, args.out, args.tag)
    os.makedirs(outdir, exist_ok=True)
    for config in model.peft_config.values():
        config.base_model_name_or_path = portable_base(ROOT, model_path)
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    saved_args = vars(args).copy()
    try:
        saved_args["data"] = os.path.relpath(args.data, ROOT).replace(os.sep, "/")
    except ValueError:
        saved_args["data"] = os.path.basename(args.data)
    meta = {"args": saved_args, "trainable_params": trainable, "total_params": total,
            "steps": step, "elapsed_sec": round(time.time() - t0, 1),
            "device": device, "base_model": args.model}
    with open(os.path.join(outdir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n完成：{step} 步，用时 {time.time()-t0:.0f}s，LoRA 保存至 {outdir}")


if __name__ == "__main__":
    main()
