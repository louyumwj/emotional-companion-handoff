"""零训练基线：给出各字段的"地板分"，供后续模型对比。

B0 多数类：情绪恒为 anxiety，画像/记忆全空，回复空串
B1 规则情绪：复用 app/emotion.py 的词典，映射到官方 16 类；其余同 B0；回复取历史最后一条助手发言
B2 规则情绪 + 关键词画像：额外用关键词抽取 interests / style / traits；其余同 B1
"""
import collections
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "eval"))
sys.path.insert(0, ROOT)

from official_scorer import main as score_main  # noqa: E402

GOLD = os.path.join(ROOT, "data", "processed", "official_val.jsonl")
OUTDIR = os.path.join(ROOT, "reports")

EMO_MAP = {
    "焦虑": "anxiety", "低落": "sadness", "愤怒": "anger", "孤独": "loneliness",
    "疲惫": "helplessness", "开心": "joy", "平静": "neutral",
}

INTEREST_KW = {
    "games": ["游戏", "打游戏", "手游", "端游", "开黑", "上分"],
    "music": ["音乐", "唱歌", "吉他", "钢琴", "耳机", "演唱会"],
    "sports_fitness": ["健身", "跑步", "篮球", "足球", "运动", "游泳", "羽毛球", "瑜伽"],
    "study_exam": ["考试", "考研", "保研", "论文", "绩点", "作业", "答辩", "课程", "导师"],
    "career_development": ["实习", "求职", "就业", "面试", "简历", "秋招", "offer", "职业"],
    "reading_writing": ["读书", "看书", "写作", "小说", "书"],
    "film_animation": ["电影", "动漫", "番剧", "追剧", "综艺"],
    "pets": ["猫", "狗", "宠物", "养猫", "养狗"],
    "travel_outdoor": ["旅行", "旅游", "露营", "爬山", "户外", "徒步"],
    "programming_technology": ["编程", "代码", "算法", "程序", "电脑", "技术"],
    "art_design": ["设计", "画画", "绘画", "摄影", "手工"],
    "social": ["聚会", "社交", "朋友", "同学", "社团", "圈子"],
}
STYLE_KW = {
    "brief": ["简单说", "简短", "长话短说"],
    "detailed": ["详细", "仔细说", "具体讲"],
    "colloquial": ["就", "呗", "啥", "咋", "挺", "老", "嗨"],
    "formal": ["您好", "请问", "非常感谢"],
    "direct": ["直接", "说白了", "我就说"],
    "indirect": ["可能", "也许", "大概", "有点", "好像"],
    "humorous": ["哈哈", "搞笑", "笑死", "段子"],
    "rational": ["分析", "理性", "逻辑", "为什么", "原因", "道理"],
    "high_emotional_expression": ["！", "啊", "太", "特别", "超级", "真的"],
    "low_emotional_expression": [],
    "emoji_user": [],
}
TRAIT_KW = {
    "sensitive": ["敏感", "在意", "想很多", "反复", "内耗", "纠结", "放不下"],
    "introverted": ["内向", "社恐", "不爱说话", "独处", "一个人"],
    "extroverted": ["外向", "爱说话", "喜欢热闹", "爱交朋友"],
    "conservative": ["保守", "传统", "父母说", "应该"],
    "high_conscientiousness": ["计划", "自律", "认真", "安排", "目标", "努力"],
    "casual": ["随便", "随性", "无所谓", "差不多"],
    "agreeable": ["不好意思", "麻烦", "谢谢", "抱歉"],
    "assertive": ["不想", "拒绝", "我决定", "我要"],
    "emotionally_stable": ["还好", "没事", "过去了", "已经接受"],
    "open": ["想尝试", "好奇", "新的", "没试过"],
}


def rule_emotion(text):
    try:
        from app.emotion import analyze
        return EMO_MAP.get(analyze(text)["emotion"], "neutral")
    except Exception:
        return "neutral"


def rule_profile(text):
    p = {k: [] for k in ("personality_traits", "interests", "style")}
    for key, kws in INTEREST_KW.items():
        if any(k in text for k in kws):
            p["interests"].append(key)
    for key, kws in STYLE_KW.items():
        if kws and any(k in text for k in kws):
            p["style"].append(key)
    for key, kws in TRAIT_KW.items():
        if any(k in text for k in kws):
            p["personality_traits"].append(key)
    return p


def user_text(inst):
    return " ".join(t["content"] for t in inst["history"] if t["role"] == "user")


def last_assistant(inst):
    for t in reversed(inst["history"]):
        if t["role"] == "assistant":
            return t["content"]
    return ""


def main():
    insts = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    os.makedirs(OUTDIR, exist_ok=True)
    variants = {
        "B0_多数类": lambda i: ("anxiety", {k: [] for k in
                                        ("personality_traits", "interests", "style")}, "", []),
        "B1_规则情绪+空画像": lambda i: (rule_emotion(user_text(i)),
                                  {k: [] for k in ("personality_traits", "interests", "style")},
                                  last_assistant(i), []),
        "B2_规则情绪+关键词画像": lambda i: (rule_emotion(user_text(i)), rule_profile(user_text(i)),
                                     last_assistant(i), []),
    }
    summaries = {}
    for name, fn in variants.items():
        path = os.path.join(OUTDIR, f"_pred_{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in insts:
                emo, prof, resp, mem = fn(i)
                f.write(json.dumps({"instance_id": i["instance_id"], "response_text": resp,
                                    "emotion_label": emo, "user_profile": prof,
                                    "memory_refs": mem}, ensure_ascii=False) + "\n")
        print("\n" + "#" * 78)
        print(f"# 基线 {name}")
        print("#" * 78)
        sys.argv = ["official_scorer", "--pred", path, "--gold", GOLD,
                    "--out", os.path.join(OUTDIR, f"baseline_{name}.md")]
        summaries[name] = score_main()

    print("\n" + "=" * 78)
    print("基线汇总")
    print("=" * 78)
    hdr = f"{'方案':<26}{'情绪Acc':>9}{'情绪F1':>9}{'画像全对':>10}{'记忆F1':>9}{'ROUGE-L':>9}"
    print(hdr)
    for k, v in summaries.items():
        print(f"{k:<26}{v['emotion_acc']:>9.4f}{v['emotion_macro_f1']:>9.4f}"
              f"{v['profile_exact']:>10.4f}{v['memory_f1']:>9.4f}{v['rouge_l']:>9.4f}")
    with open(os.path.join(OUTDIR, "baselines_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
