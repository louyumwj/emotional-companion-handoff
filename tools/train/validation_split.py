"""Deterministic conversation-level calibration/holdout split."""
import random


def split_validation(rows, seed=42):
    groups = [r.get("conversation_id") or r["instance_id"].split("#", 1)[0] for r in rows]
    conversations = sorted(set(groups))
    if len(conversations) < 2:
        raise ValueError("At least two conversations are required for calibration/holdout")
    random.Random(seed).shuffle(conversations)
    calib_groups = set(conversations[:len(conversations) // 2])
    calib = [i for i, group in enumerate(groups) if group in calib_groups]
    holdout = [i for i, group in enumerate(groups) if group not in calib_groups]
    return calib, holdout
