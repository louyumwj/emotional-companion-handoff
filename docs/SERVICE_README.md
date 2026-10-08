# 数字人综合情感陪伴对话模型 · 可提交镜像骨架
> 此文是早期自由对话服务说明。继续开发以 `HANDOFF.md` 和 `FIXES_20261008.md` 为准；四字段评测接口仍属于 T5 待办。

2026 动感地带AI+高校创智计划 · AI技术赛道 · 命题方向二
交付形态：**评测平台拉起容器 → 调用 HTTP 接口 → 自动打分**

---

## 1. 架构（感知 — 评估 — 干预 闭环）

```
用户消息
   │
   ├─▶ 感知  app/emotion.py   情绪(7类) + 强度 + 压力源话题
   ├─▶ 评估  app/safety.py    风险分级 0~3（含自伤/轻生）
   │
   ├─▶ 长记忆 app/memory.py
   │      · 多轮历史窗口（HISTORY_TURNS）
   │      · 向量召回相关经历（无向量模型时退化为字符 n-gram）
   │      · 结构化用户画像（主导情绪 / 话题 / 近期趋势）
   │
   ├─▶ 干预  app/empathy.py   8 类共情策略（ESConv 体系）→ system prompt
   │         风险 3 → 固定转介话术（不经过模型）
   │
   └─▶ 生成  app/llm.py       OpenAI 兼容后端（LMDeploy / vLLM / Ollama）
              └─ 失败 → 规则兜底话术（服务永不 500）
```

**为什么这样分层**：赛题三个关键词「情感识别 / 长记忆对话 / 共情反馈」正好对应感知、记忆、干预三层，
答辩时可以逐层指认技术实现，而不是只说「我们用了大模型」。

---

## 2. 快速开始

```bash
# 构建
docker build -t emotional-companion:0.1 .

# 有 GPU（模型权重放在 ./models/qwen2.5-7b-awq）
docker run --gpus all -p 8000:8000 -v "%cd%/models:/workspace/models" emotional-companion:0.1

# 无 GPU / 只想验证接口契约
docker run -p 8000:8000 -e LLM_BACKEND=none emotional-companion:0.1
```

自测：

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id":"s1","user_id":"u1","message":"下周答辩，我什么都做不出来，整夜失眠"}'

curl -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b-awq","messages":[{"role":"user","content":"最近没人懂我"}]}'
```

---

## 3. 接口契约

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 存活 + **模型是否真的就绪**（`llm_ready`） |
| GET | `/v1/models` | OpenAI 兼容探测 |
| POST | `/v1/chat/completions` | OpenAI 兼容对话（支持 `stream=true`，SSE） |
| POST | `/chat` | 简易契约，直接返回情绪/风险/策略等调试信息 |
| POST | `/session/reset` | 重置某个会话/用户的记忆（评测重置用） |
| GET | `/stats` | 请求数、走模型/兜底/危机次数 |

不带会话或用户标识的请求按无状态处理，不保存历史和用户记忆。需要持续记忆时，`/chat` 显式提供 `session_id`/`user_id`，或 OpenAI 请求提供 `user`/`X-Session-Id`。显式空历史 `[]` 会覆盖会话历史；`/session/reset` 清理关联用户记忆及会话。调用者传入的 system 历史均被过滤。

`/chat` 返回示例：

```json
{
  "reply": "听起来这件事一直悬在你心里……",
  "emotion": {"emotion": "焦虑", "intensity": 0.7, "topics": ["学业", "身心"], "signals": ["失眠", "慌"]},
  "risk": 2,
  "strategies": ["Reflection of feelings", "Affirmation and Reassurance", "Question"],
  "profile": {"turns": 3, "dominant_emotion": "焦虑", "top_topics": ["学业"], "recent_trend": ["焦虑", "焦虑", "低落"]},
  "route": "llm"
}
```

> 评测方如果用的是自定义契约（例如 `{"query": "..."} → {"response": "..."}`），
> 只需在 `app/server.py` 加一个薄适配层，核心 `generate()` 不用改。

---

## 4. 关键环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_BACKEND` | `openai` | `openai` 走本机推理服务；`none` 纯规则兜底 |
| `MODEL_DIR` | `/workspace/models/qwen2.5-7b-awq` | LMDeploy 加载的权重目录 |
| `MODEL_FORMAT` | `awq` | `awq` / `hf` / `gguf` |
| `LLM_BASE_URL` | `http://127.0.0.1:23333/v1` | OpenAI 兼容地址 |
| `MAX_NEW_TOKENS` | `256` | 陪伴回复不需要长，短回复更稳更省时 |
| `REQUEST_TIMEOUT` | `30` | 单次生成超时（秒） |
| `HISTORY_TURNS` | `12` | 保留多少轮对话进 prompt |
| `MEMORY_TOP_K` | `4` | 每轮召回多少条相关经历 |
| `CRISIS_HOTLINE` | `12356` | 转介热线（**按最新官方信息核对**） |
| `STATE_DIR` | `/workspace/state` | 记忆持久化目录 |

---

## 5. EmoLLM 怎么用（重要）

**能用，但不要直接交它的权重。** 三层原因：

1. **合规与原创**：赛事要求作品原创、核心开发在赛期内独立完成。直接提交他人权重在「创新性」上吃亏。
   正确姿势是 **用 EmoLLM 的数据与训练流程（SFT 脚本、心理对话语料、评测集、RAG 方案）自训 LoRA**，
   基座优先选 **Qwen2.5-7B-Instruct（Apache-2.0）**，其次 InternLM2.5-7B-chat（许可须自行核对最新条款）。
2. **体积**：7B fp16 ≈ 15GB；AWQ/W4A16 ≈ 5GB；GGUF Q4_K_M ≈ 4.7GB。私有镜像配额决定生死。
3. **卡时**：初赛 20 卡时，全参微调不可能，**只能 LoRA**。

推荐组合：

| 模块 | 选型 | 体积 |
|---|---|---|
| 对话生成 | Qwen2.5-7B-Instruct + AWQ（或 EmoLLM 数据 LoRA 后量化） | ≈5GB |
| 长记忆向量 | bge-small-zh-v1.5（可选） | ≈100MB |
| 情感识别 | 本骨架内置规则实现（0 依赖）；需要更强再接 Chinese-Emotion-Small ONNX | 0~100MB |
| 兜底 | 规则话术 | 0 |

显存不够就把基座降到 Qwen2.5-3B / 1.5B，接口层完全不用改。

---

## 6. 20 卡时预算

| 事项 | 卡时 | 备注 |
|---|---|---|
| 语料构造 / 清洗 / 策略标注（CPU） | 0 | 用 EmoLLM 数据 + ESConv 策略体系 |
| LoRA SFT（7B，1~2 epoch） | 5~7 | 先本地小模型跑通再上平台 |
| 自动评测 + 消融 | 2~3 | 共情策略有无 / 长记忆有无，各跑一轮 |
| 联调 + 演示视频录制（推理也吃卡） | 8~10 | 留足，别到时候没卡录视频 |

---

## 7. 提交前 Checklist

- [ ] 镜像**不依赖外网**：`pip` 装在构建期完成，运行时 `HF_HUB_OFFLINE=1`
- [ ] 权重**已在镜像内**（评测平台不会给你挂载数据卷）
- [ ] `/health` 在模型未就绪时也返回 200，但 `llm_ready=false`（避免被判定启动失败）
- [ ] 冷启动时间 < 平台超时（LMDeploy 加载 7B 约 1~3 分钟，`--start-period=300s`）
- [ ] 断网 / 后端崩溃 → 自动降级规则模式，服务仍可对话
- [ ] 危机场景实测：`"不想活了"` 必须返回转介话术
- [ ] 多轮实测 10 轮以上，确认不重复、不忘事、不越界诊断
- [ ] 镜像里没有 `.git`、数据集、checkpoint 等无关大文件（见 `.dockerignore`）

---

## 8. 已知限制

- 记忆默认存在容器文件系统（`STATE_DIR`），**容器重建即丢失**；若评测要求跨会话长期记忆，需要接入外部存储或平台提供的持久化卷。
- 词典情感识别对反讽、方言、长文本效果有限，接口已预留替换点（`app/emotion.py: analyze`）。
- `stream=true` 目前一次性返回整段（合规客户端可用），如需真流式可在 `llm.py` 加 `stream=True` 透传。
- Docker 镜像未在本机构建验证（本机未安装 Docker），代码已通过 `py_compile` 与模块级冒烟测试。
