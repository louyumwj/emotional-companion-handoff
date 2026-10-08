"""用 SFT 模型生成回复，并组装成官方四字段预测文件。

可用 --t2-pred 注入 T2 的情绪/画像预测（同时写进 system prompt 作为生成条件）。

用法：
  python tools/train/infer_sft.py --model models/Qwen2.5-0.5B-Instruct \
      --adapter models/t3_lora/smoke --data data/processed/official_val.jsonl \
      --limit 30 --out reports/t3_pred_smoke.jsonl
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_SYSTEM = (
    "你是面向中国大学生的情感陪伴伙伴。请先准确接住对方的情绪，再决定是否提供建议；"
    "语气自然、口语化，一次回复 2~4 句；不做诊断、不开药、不下病理结论。"
)
HINT_TMPL = "\n【参考信息】对方当前情绪倾向：{emotion}；画像线索：{profile}。请自然融入，不要直接复述。"


def conditioned_system(emotion, profile):
    flat = [v for vs in profile.values() for v in vs]
    return BASE_SYSTEM + HINT_TMPL.format(emotion=emotion, profile="、".join(flat) or "未提供")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--data", default="data/processed/official_val.jsonl")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--t2-pred", default="", help="T2 预测文件，用于注入情绪/画像条件")
    ap.add_argument("--out", default="reports/t3_pred.jsonl")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.max_input_tokens < 16 or args.max_new_tokens < 1 or args.limit < 0:
        ap.error("Token budgets must be positive; --max-input-tokens >= 16 and --limit >= 0")

    if args.threads:
        torch.set_num_threads(args.threads)
    model_path = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Match SFT's suffix-preserving prompt crop: keep the latest user turn and generation marker.
    tok.truncation_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype,
                                                 trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel
        apath = args.adapter if os.path.isabs(args.adapter) else os.path.join(ROOT, args.adapter)
        model = PeftModel.from_pretrained(model, apath)
        print(f"已加载 LoRA 适配器：{apath}", flush=True)
    model.to(device).eval()

    insts = [json.loads(l) for l in open(os.path.join(ROOT, args.data), encoding="utf-8")]
    if args.limit:
        insts = insts[:args.limit]
    t2 = {}
    if args.t2_pred:
        for line in open(os.path.join(ROOT, args.t2_pred), encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                t2[r["instance_id"]] = r
        missing = [r["instance_id"] for r in insts if r["instance_id"] not in t2]
        if missing:
            raise ValueError(f"T2 predictions missing for {len(missing)} instances: {missing[:3]}")

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f, torch.no_grad():
        for i, inst in enumerate(insts, 1):
            system = BASE_SYSTEM
            emo_out, prof_out, mem_out = "neutral", {k: [] for k in
                                                     ("personality_traits", "interests", "style")}, []
            if inst["instance_id"] in t2:
                t = t2[inst["instance_id"]]
                emo_out = t.get("emotion_label", "neutral")
                prof_out = t.get("user_profile", prof_out)
                system = conditioned_system(emo_out, prof_out)
            msgs = ([{"role": "system", "content": system}]
                    + [{"role": t["role"], "content": t["content"]} for t in inst["history"]])
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False,
                      truncation=True, max_length=args.max_input_tokens).to(device)
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 repetition_penalty=1.05,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            reply = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            f.write(json.dumps({"instance_id": inst["instance_id"], "response_text": reply,
                                "emotion_label": emo_out, "user_profile": prof_out,
                                "memory_refs": mem_out}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 5 == 0 or i == 1:
                el = time.time() - t0
                print(f"  {i}/{len(insts)}  {el:.0f}s  ({el/i:.1f}s/条)", flush=True)
    print(f"\n生成完成：{len(insts)} 条，用时 {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
