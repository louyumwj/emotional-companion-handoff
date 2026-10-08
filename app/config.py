"""全部配置走环境变量，方便评测平台不改代码直接调参。"""
import os
from dataclasses import dataclass, field


def _s(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    port: int = _i("PORT", 8000)
    # 推理后端：openai(走本机 LMDeploy/vLLM 的 OpenAI 接口) | none(纯规则兜底)
    llm_backend: str = _s("LLM_BACKEND", "openai")
    llm_base_url: str = _s("LLM_BASE_URL", "http://127.0.0.1:23333/v1")
    llm_model: str = _s("LLM_MODEL", "qwen2.5-7b-awq")
    llm_api_key: str = _s("LLM_API_KEY", "EMPTY")
    max_new_tokens: int = _i("MAX_NEW_TOKENS", 256)
    temperature: float = _f("TEMPERATURE", 0.7)
    top_p: float = _f("TOP_P", 0.9)
    request_timeout: float = _f("REQUEST_TIMEOUT", 30.0)
    # 多轮与长记忆
    history_turns: int = _i("HISTORY_TURNS", 12)
    memory_top_k: int = _i("MEMORY_TOP_K", 4)
    state_dir: str = _s("STATE_DIR", "/workspace/state")
    embed_model_dir: str = _s("EMBED_MODEL_DIR", "/workspace/models/bge-small-zh-v1.5")
    # 合规
    crisis_hotline: str = _s("CRISIS_HOTLINE", "12356")
    disclaimer: str = _s(
        "DISCLAIMER",
        "我是AI情感陪伴助手，不能替代专业心理咨询与医疗诊断。",
    )
    extra: dict = field(default_factory=dict)


settings = Settings()
