"""长记忆：多轮对话历史 + 用户画像 + 相关经历召回。

赛题明确要求「长记忆对话」，所以这一层是评分关键，不能只靠把历史塞进 prompt：
1. 全量历史 → prompt 爆炸、超长截断后前后矛盾；
2. 这里做「历史窗口 + 向量召回 + 结构化画像」三件套；
3. 向量模型不可用时自动退化为字符 bigram 相似度，保证离线可用。
"""
import json
import math
import os
import time
from collections import Counter

from .config import settings

try:  # 可选依赖，缺了也能跑
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


def _bigrams(text: str) -> set[str]:
    text = "".join(text.split())
    return {text[i:i + 2] for i in range(max(0, len(text) - 1))}


class MemoryStore:
    def __init__(self) -> None:
        self.users: dict[str, list[dict]] = {}
        self._embedder = None
        self._embed_tried = False
        self._state_path = os.path.join(settings.state_dir, "memory.json")
        self._load()

    # ---------- 向量模型（懒加载，失败即退化） ----------
    @property
    def embedder(self):
        if not self._embed_tried:
            self._embed_tried = True
            if SentenceTransformer is not None and os.path.isdir(settings.embed_model_dir):
                try:
                    self._embedder = SentenceTransformer(settings.embed_model_dir)
                except Exception as exc:  # pragma: no cover
                    print(f"[memory] 向量模型加载失败，退化为 n-gram 检索: {exc}", flush=True)
        return self._embedder

    def _vector(self, text: str):
        emb = self.embedder
        if emb is None:
            return None
        try:
            return emb.encode(text, normalize_embeddings=True).tolist()
        except Exception:  # pragma: no cover
            return None

    # ---------- 写入 ----------
    def add(self, user_id: str, text: str, emotion: dict, reply: str = "") -> None:
        if not user_id or not text:
            return
        item = {
            "text": text,
            "reply": reply,
            "emotion": emotion.get("emotion"),
            "intensity": emotion.get("intensity", 0.0),
            "topics": emotion.get("topics", []),
            "ts": time.time(),
            "vec": self._vector(text),
        }
        self.users.setdefault(user_id, []).append(item)
        # 控制内存与文件体积
        if len(self.users[user_id]) > 500:
            self.users[user_id] = self.users[user_id][-500:]
        self._save()

    # ---------- 召回 ----------
    def retrieve(self, user_id: str, query: str, k: int | None = None) -> list[dict]:
        k = k or settings.memory_top_k
        items = self.users.get(user_id, [])
        if not items:
            return []

        qvec = self._vector(query)
        scored = []
        if qvec is not None:
            for it in reversed(items[-200:]):
                if not it.get("vec"):
                    continue
                sim = sum(a * b for a, b in zip(qvec, it["vec"]))
                scored.append((sim, it))
        else:
            qset = _bigrams(query)
            for it in reversed(items[-200:]):
                iset = _bigrams(it["text"])
                if not qset or not iset:
                    continue
                sim = len(qset & iset) / math.sqrt(len(qset) * len(iset))
                scored.append((sim, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"text": it["text"], "emotion": it["emotion"], "topics": it.get("topics", []),
                 "score": round(float(s), 3)} for s, it in scored[:k] if s > 0.15]

    # ---------- 画像 ----------
    def profile(self, user_id: str) -> dict:
        items = self.users.get(user_id, [])
        if not items:
            return {"turns": 0}
        emotions = Counter(i["emotion"] for i in items if i.get("emotion"))
        topics = Counter(t for i in items for t in i.get("topics", []))
        recent = items[-5:]
        trend = [i["emotion"] for i in recent]
        avg_intensity = round(sum(i.get("intensity", 0) for i in recent) / max(1, len(recent)), 2)
        return {
            "turns": len(items),
            "dominant_emotion": emotions.most_common(1)[0][0] if emotions else None,
            "emotion_counts": dict(emotions.most_common(5)),
            "top_topics": [t for t, _ in topics.most_common(3)],
            "recent_trend": trend,
            "recent_intensity": avg_intensity,
        }

    # ---------- 持久化（容器内有效；跨容器需外部存储） ----------
    def _save(self) -> None:
        try:
            os.makedirs(settings.state_dir, exist_ok=True)
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(self.users, f, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover
            print(f"[memory] 持久化失败（不影响服务）: {exc}", flush=True)

    def _load(self) -> None:
        try:
            if os.path.isfile(self._state_path):
                with open(self._state_path, encoding="utf-8") as f:
                    self.users = json.load(f)
                print(f"[memory] 载入 {sum(len(v) for v in self.users.values())} 条历史", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[memory] 历史载入失败: {exc}", flush=True)

    def reset(self, user_id: str) -> None:
        self.users.pop(user_id, None)
        self._save()


store = MemoryStore()
