"""安全护栏：危机识别 + 转介。

评测对心理健康类赛题几乎一定会查「风险场景是否被正确处理」，
所以这一层优先级高于对话质量，任何情况下都不能被绕过或吞掉。
"""
import re

from .config import settings

# 3 级：明确的自伤/轻生意图，必须立刻转介
CRISIS_L3 = [
    "自杀", "轻生", "不想活", "想死", "去死", "活不下去", "结束生命", "了结自己",
    "割腕", "跳楼", "上吊", "烧炭", "自残", "伤害自己",
]

# Action and quantity matter for medication; a prescribed drug name alone is not intent.
CRISIS_ACTIONS = [
    re.compile(r"(?:吞|吃|服|喝)(?:下|了|掉|用)?[^，。！？；\n]{0,8}(?:一整瓶|整瓶|一瓶|一把|一大把|大量|过量)[^，。！？；\n]{0,4}(?:药|安眠|镇静|农药)"),
    re.compile(r"(?:从|在)[^，。！？；\n]{0,10}(?:天台|楼顶|屋顶|高楼|桥上|桥边)[^，。！？；\n]{0,8}(?:跳下|跳出去|跳进)"),
    re.compile(r"(?:想|准备|打算|决定|要)[^，。！？；\n]{0,5}(?:跳下去|跳河|跳海|跳轨|卧轨)"),
]
NEGATED_INTENT = re.compile(
    r"(?:不|没|没有|不想|不会|不打算|不准备|没有想|没想|没有打算|并不想|不曾|从未|不要|别|劝[^，。！？；]{0,4}不要)\s*(?:再|去|要)?\s*$"
)


def _affirmed_hits(text):
    hits = []
    clauses = re.split(r"[，。！？；\n]|但是|但|不过", text)
    for clause in clauses:
        for word in CRISIS_L3:
            for match in re.finditer(re.escape(word), clause):
                if not NEGATED_INTENT.search(clause[:match.start()]):
                    hits.append(word)
        for pattern in CRISIS_ACTIONS:
            for match in pattern.finditer(clause):
                if not NEGATED_INTENT.search(clause[:match.start()]):
                    hits.append(match.group())
    return list(dict.fromkeys(hits))

# 2 级：强烈痛苦 / 无意义感 / 求助信号，需要高共情 + 温和探询
CRISIS_L2 = [
    "绝望", "没意义", "没有意义", "撑不住", "撑不下去", "熬不住", "熬不下去",
    "扛不住", "扛不下去", "崩溃", "活着好累", "很累很累", "没人需要我", "消失",
    "解脱", "抑郁", "睡不着", "整夜失眠", "没有出路", "一无是处", "废物",
]

# 1 级：持续负面情绪
CRISIS_L1 = ["难受", "痛苦", "压抑", "喘不过气", "委屈", "无助", "孤独", "焦虑", "害怕"]


def assess_risk(text: str) -> tuple[int, list[str]]:
    """返回 (风险等级 0-3, 命中的信号词)。"""
    if not text:
        return 0, []
    hits3 = _affirmed_hits(text)
    if hits3:
        return 3, hits3
    hits2 = [w for w in CRISIS_L2 if w in text]
    if hits2:
        return 2, hits2
    hits1 = [w for w in CRISIS_L1 if w in text]
    if hits1:
        return 1, hits1
    return 0, []


def crisis_message(risk: int) -> str:
    """风险等级 3 时的固定转介话术（不交给模型自由发挥）。"""
    hotline = settings.crisis_hotline
    return (
        "我听到你现在非常痛苦，谢谢你愿意说出来，你的感受很重要，我会陪着你。\n"
        "但我必须坦诚：我无法提供你此刻需要的专业支持。请你现在就联系能真正帮到你的人——\n"
        f"· 全国心理援助热线 {hotline}（24 小时，免费）\n"
        "· 学校心理健康教育与咨询中心（工作日可直接前往或电话预约）\n"
        "· 如果情况紧急，请拨打 120，或立刻告诉身边的老师、室友、家人，让他们陪着你。\n"
        "你并不孤单，这件事不需要你一个人扛。可以答应我去打一个电话吗？"
    )
