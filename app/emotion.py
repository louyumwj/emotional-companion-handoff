"""情感识别：词典加权 + 否定处理 + 压力源抽取。

为什么先做规则而不是直接上模型：
1. 镜像体积与显存都要省（初赛只有 20 卡时，显卡要留给对话模型）；
2. 规则完全可解释，答辩时能讲清「情感识别是怎么做的」，比黑盒调 API 更抗问；
3. 后续可无缝替换为 Chinese-Emotion-Small / Qwen 分类，接口保持不变。
"""
import re

EMOTIONS = ["焦虑", "低落", "愤怒", "孤独", "疲惫", "开心", "平静"]

LEXICON: dict[str, dict[str, float]] = {
    "焦虑": {
        "焦虑": 2, "紧张": 2, "慌": 2, "担心": 1.5, "害怕": 2, "不安": 2, "忐忑": 2,
        "压力大": 2, "心慌": 2, "睡不着": 1.5, "失眠": 1.5, "来不及": 1.5, "怕": 1.5,
        "怎么办": 1, "来不及了": 2, "心里没底": 2,
    },
    "低落": {
        "难过": 2, "伤心": 2, "低落": 2, "沮丧": 2, "失落": 2, "想哭": 2.5, "哭": 1.5,
        "没意思": 1.5, "没意义": 2.5, "绝望": 3, "抑郁": 3, "空虚": 2, "提不起劲": 2,
        "一无是处": 3, "废物": 2.5, "失败": 2,
        # 危机相关（同时会被 safety 层命中，这里是为了让画像也记得住）
        "不想活": 3, "活不下去": 3, "想死": 3, "轻生": 3,
        # 注意：「没用」单独出现极易误判（如"说了也没用"），必须带主语
        "我真没用": 2.5, "我很没用": 2.5, "自己没用": 2.5, "我太没用了": 2.5,
    },
    "愤怒": {
        "生气": 2, "愤怒": 2.5, "气死": 2.5, "烦": 1.5, "讨厌": 2, "恶心": 2,
        "凭什么": 2, "受不了": 2, "不公平": 2, "火大": 2.5, "憋屈": 2, "委屈": 1.5,
        "烦死": 2.5, "烦人": 2, "气人": 2, "恼火": 2, "无语": 1.5,
    },
    "孤独": {
        "孤独": 2.5, "孤单": 2.5, "没人": 2, "一个人": 1.5, "没朋友": 2.5,
        "融不进": 2, "被孤立": 2.5, "没人懂": 2.5, "没人理解": 2.5, "格格不入": 2,
    },
    "疲惫": {
        "累": 1.5, "疲惫": 2, "好累": 2, "撑不住": 2.5, "扛不住": 2.5, "崩溃": 2.5,
        "压力": 1.5, "内耗": 2, "麻木": 2, "不想动": 1.5, "熬夜": 1.5, "忙": 1,
    },
    "开心": {
        "开心": 2, "高兴": 2, "快乐": 2, "太好了": 2, "兴奋": 2, "满足": 1.5,
        "谢谢": 1, "喜欢": 1.5, "顺利": 1.5, "赢了": 1.5, "进步": 1.5, "有希望": 2,
    },
}

# 压力源 / 话题（写进用户画像，让长记忆有内容可记）
TOPICS: dict[str, list[str]] = {
    "学业": ["考试", "期末", "考研", "保研", "论文", "开题", "答辩", "挂科", "绩点", "作业", "成绩"],
    "就业": ["求职", "找工作", "面试", "实习", "简历", "秋招", "春招", "offer", "裁员", "考公"],
    "人际": ["室友", "同学", "朋友", "舍友", "闹矛盾", "吵架", "社团", "导师", "被排挤"],
    "家庭": ["父母", "家里", "爸妈", "家人", "亲戚", "回家"],
    "恋爱": ["分手", "恋爱", "喜欢的人", "异地", "暗恋", "失恋", "对象"],
    "经济": ["没钱", "生活费", "兼职", "学费", "欠", "花呗"],
    "身心": ["失眠", "睡不着", "头疼", "胃", "身体", "生病", "熬夜", "减肥", "胖"],
    "自我": ["自卑", "不够好", "别人都", "比较", "迷茫", "不知道未来", "方向"],
}

NEGATION = ("不", "没", "别", "无", "毫无")


def _negated(text: str, idx: int) -> bool:
    window = text[max(0, idx - 3):idx]
    return any(n in window for n in NEGATION)


def extract_topics(text: str) -> list[str]:
    return [t for t, kws in TOPICS.items() if any(k in text for k in kws)]


def analyze(text: str) -> dict:
    """返回情感识别结果。

    {
      "emotion": "焦虑",
      "intensity": 0.0~1.0,
      "scores": {...},
      "topics": ["学业"],
      "signals": ["考试", "睡不着"]
    }
    """
    text = (text or "").strip()
    if not text:
        return {"emotion": "平静", "intensity": 0.0, "scores": {}, "topics": [], "signals": []}

    scores = {e: 0.0 for e in EMOTIONS}
    signals: list[str] = []
    for emo, words in LEXICON.items():
        for w, weight in words.items():
            start = 0
            while True:
                idx = text.find(w, start)
                if idx < 0:
                    break
                if _negated(text, idx) and emo == "开心":
                    scores["低落"] += weight * 0.4
                elif _negated(text, idx) and emo in ("焦虑", "疲惫"):
                    scores["平静"] += weight * 0.3
                else:
                    scores[emo] += weight
                    signals.append(w)
                start = idx + len(w)

    # 标点与长度带来的情绪强度加成
    if re.search(r"[!！]{1,}", text):
        for e in ("愤怒", "开心", "焦虑"):
            scores[e] *= 1.15
    if "?" in text or "？" in text:
        scores["焦虑"] *= 1.05

    top = max(scores, key=lambda k: scores[k]) if scores else "平静"
    top_score = scores.get(top, 0.0)
    if top_score <= 0:
        top, top_score = "平静", 0.6
    intensity = round(min(1.0, top_score / 5.0), 2)

    return {
        "emotion": top,
        "intensity": intensity,
        "scores": {k: round(v, 2) for k, v in scores.items() if v > 0},
        "topics": extract_topics(text),
        "signals": sorted(set(signals))[:8],
    }
