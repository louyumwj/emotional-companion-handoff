"""官方标签体系（唯一真源，训练/评测/部署共用）。

来源：官方 schema（val_public.schema.json）的 enum 定义。
"""

EMOTIONS = [
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise", "mixed",
    "sadness", "loneliness", "anxiety", "anger", "fear", "disgust", "shame", "helplessness",
]

TRAITS = [
    "extroverted", "introverted", "open", "conservative", "high_conscientiousness",
    "casual", "agreeable", "assertive", "emotionally_stable", "sensitive",
]

INTERESTS = [
    "study_exam", "programming_technology", "reading_writing", "film_animation", "music",
    "games", "sports_fitness", "travel_outdoor", "pets", "social", "career_development",
    "art_design",
]

STYLES = [
    "brief", "detailed", "colloquial", "formal", "direct", "indirect", "humorous",
    "rational", "high_emotional_expression", "low_emotional_expression", "emoji_user",
]

PROFILE_GROUPS = {"personality_traits": TRAITS, "interests": INTERESTS, "style": STYLES}

# 中文显示名，仅用于报告
EMOTION_ZH = {
    "joy": "开心", "gratitude": "感激", "relaxed": "放松", "care": "关心", "pride": "自豪",
    "neutral": "平静", "surprise": "惊讶", "mixed": "复杂", "sadness": "悲伤",
    "loneliness": "孤独", "anxiety": "焦虑", "anger": "愤怒", "fear": "恐惧",
    "disgust": "厌恶", "shame": "羞耻", "helplessness": "无力",
}
