# 交接说明（HANDOFF）

> 历史交接文档。当前任务、执行步骤和验收以 `NEXT_STEPS.md` 为准；T3 正式训练、T4 和 T5 尚未完成。GPU 卡时/显存仅为估算。

> 2026 动感地带AI+高校创智计划 · AI技术赛道 · 命题二「数字人综合情感陪伴对话模型」
> 最后更新：2026-10-08　｜　**请先读完本文再动手**
> 上一阶段：T1–T3 已完成（详见 `reports/T2T3_完成报告.md`）

> 修正版说明（2026-10-08）：见 `FIXES_20261008.md`。已修复路径迁移、画像计分、推理截断、会话隔离和复现错误；最新校准结果以 `reports/T2_报告.md` 为准，完成报告中的旧分数是历史记录。模型未重新训练。

---

## 0. 30 秒速览

**这是一个什么任务**：不是自由对话，是**结构化多任务预测**。给定一段对话历史，在每个"预测点"输出四个字段：

| 字段 | 类型 | 现状 |
|---|---|---|
| `response_text` | 文本（助手该说什么） | 管线已通，**7B 全量训练待 GPU** |
| `emotion_label` | 16 分类 | ✅ 管线可用；最新分组 holdout Acc 0.7279 / Macro-F1 0.6369（旧检查点的诊断分数） |
| `user_profile` | 3 组多标签 | ❌ 未达标，已诊断出「信号不足」，建议默认输出空 |
| `memory_refs` | 记忆 ID 列表 | ⛔ **阻塞**：官方未提供记忆库，无法预测 |

**你现在最该做的两件事**：
1. **去问组委会三个问题**（见 §7），其中记忆库那个直接决定 1/4 的分能不能拿；
2. **把 T3 的 7B 训练跑起来**（按 `docs/T3_训练运行手册.md`，约 5–7 卡时）。

---

## 1. 一分钟环境复原

```powershell
# 前提：Windows + Python 3.11+（原项目用 anaconda 的 3.13）
pwsh -File setup.ps1
pwsh -File setup.ps1 -SkipModels     # 若只想先看代码/文档
```
`setup.ps1` 会：建虚拟环境 `.venv-data` → 装依赖（走清华源）→ 下载两个基座模型（走 ModelScope）。

**包内不含**（约 2.5 GB，全部可再生，见下表和 §4.1）：

| 不包含 | 体积 | 再生方式 |
|---|---|---|
| `.venv-data/` | 825 MB | `setup.ps1` |
| `models/chinese-roberta-wwm-ext/` | 393 MB | `setup.ps1`（ModelScope） |
| `models/Qwen2.5-0.5B-Instruct/` | 953 MB | `setup.ps1`（ModelScope） |
| `data/processed/views/` | 195 MB | `build_official_dataset.py`（约 2 分钟） |
| `data/processed/official_*.jsonl` | 106 MB | 同上 |
| `data/incoming/训练-验证-数据集.zip` | 22 MB | 解压版已在包内 |

**所以接手后的第一条命令是重建训练数据**（见 §4.1）。

完成后自检：
```powershell
.venv-data\Scripts\python.exe tools\train\verify_models.py
```

---

## 2. 关键背景（不看会踩坑）

### 2.1 ⚠️ 官方公开数据的「答案泄漏」
官方给的 `turns[]` **已经包含**待预测的那条助手回复（train 18,657/18,657 条与 `target.response_text` 完全一致）。
**推理时该轮会被抹掉。** 所以：
- 构造训练数据**必须截断到 `target_user_turn_id`**（`tools/data/build_official_dataset.py` 已处理）；
- **70% 的预测点之后还有对话内容**，也必须切掉，否则模型学到"抄下文"。

### 2.2 任务结构
- `target_user_turn_id` **恒为奇数**（用户轮），回复在 `+1`；有 4 条异常数据已剔除
- 589 个预测点落在第 1 轮（**完全无历史**的冷启动）
- 官方标签体系（16 情绪 / 10 特质 / 12 兴趣 / 11 风格）**唯一真源**在 `tools/train/labels.py`

### 2.3 数据规模
train 8,338 会话 / **18,653 个预测点**；val 900 会话 / **2,354 个预测点**。

---

## 3. 目录导航

```
├── HANDOFF.md                  ← 本文
├── setup.ps1                   ← 一键环境复原
├── README.md                   ← 早期镜像骨架说明（部分已过时，见 §6 注意事项）
├── Dockerfile / entrypoint.sh  ← 镜像骨架（T5 要改造，见 §6）
├── app/                        ← 服务层模块（情绪/记忆/共情/安全/服务）
├── docs/
│   ├── 待确认清单.md            ← 需要向组委会确认的问题（含新增的记忆库问题）
│   └── T3_训练运行手册.md       ★ GPU 上训 7B 的执行依据
├── tools/
│   ├── data/                   ← 官方数据解析、实例构建
│   │   ├── analyze_official.py / analyze_official2.py / analyze_memory.py
│   │   ├── build_official_dataset.py   ★ 生成训练用实例
│   │   ├── fetch_sources.py    ← 早期开源语料下载（现已不需要）
│   │   └── inspect_zip.py
│   ├── eval/                   ★ 评测相关
│   │   ├── official_scorer.py  ★ 官方四字段评分器（改模型后必跑）
│   │   └── baselines.py        ← 零训练基线 B0/B1/B2
│   └── train/                  ★ 训练相关
│       ├── labels.py           ★ 标签体系唯一真源
│       ├── train_multitask.py  ★ T2：情绪+画像多任务
│       ├── calibrate_t2.py     ★ T2：先验校正 + 阈值校准（两工作点）
│       ├── diagnose_profile.py ← 画像头 AUC 诊断
│       ├── train_sft.py        ★ T3：Qwen LoRA SFT
│       ├── infer_sft.py        ★ T3：生成 + 官方格式组装
│       ├── download.py / download_ms.py  ← 下载器（ms=ModelScope，本机首选）
│       └── verify_models.py
├── data/
│   ├── incoming/               ← 官方原始数据 + 结构分析结果
│   │   ├── 训练-验证-数据集/     ★ 官方 train/val JSONL + schema
│   │   ├── _analysis.json      ← 全量结构统计
│   │   └── _sample_records.txt ← 2 条完整样本（含预测目标）
│   └── processed/
│       ├── DATACARD.md         ★★ 数据卡：任务定义、分布、风险、阻塞项
│       ├── official_train.jsonl ★ 18,653 条规范化实例（可直接训练）
│       ├── official_val.jsonl   ★ 2,354 条
│       ├── build_stats.json
│       └── views/              ← 训练视图（可由 build 脚本重建）
├── models/
│   ├── t2_multitask/best.pt    ★ T2 训练好的权重（元数据已迁移，张量不变）
│   ├── t2_multitask/calib.json ← 先验温度 + 逐标签阈值 + 工作点B阈值
│   └── t3_lora/smoke/          ← T3 冒烟 LoRA（未充分训练，仅证明管线）
└── reports/                    ★ 所有报告与预测文件
    ├── T2T3_完成报告.md         ★★ 先读这个
    ├── T2_报告.md / t2_diagnosis.json
    ├── T3_冒烟报告.md
    ├── baseline_B*.md          ← 零训练基线（地板分）
    └── t2_pred_val*.jsonl      ← 两个工作点的预测文件
```

`★` = 重要，`★★` = 必读。

---

## 4. 复现命令（按顺序）

### 4.1 【接手第一步】重建训练数据（包内未含，约 2 分钟）
```powershell
.venv-data\Scripts\python.exe tools\data\build_official_dataset.py
# 产出 official_{train,val}.jsonl + views/{split}_{resp_sft,cls}.jsonl
# 预期输出：train 18653 条 / val 2354 条
```

### 4.2 T2：情绪 + 画像多任务训练（CPU 约 80 分钟）
```powershell
.venv-data\Scripts\python.exe tools\train\train_multitask.py `
  --epochs 1 --batch-size 16 --max-length 256 --freeze-layers 6 --lr 3e-5 `
  --threads 16 --tag v1 --out models/t2_multitask
```
**有 GPU 的话**：去掉 `--freeze-layers`，`--batch-size 32`，几分钟即可，效果更好。

### 4.3 T2：校准 + 出报告
```powershell
.venv-data\Scripts\python.exe tools\train\calibrate_t2.py --threads 20
# 产出 reports/T2_报告.md + t2_pred_val.jsonl + t2_pred_val_exactmode.jsonl
```

### 4.4 T3：LoRA SFT
```powershell
# 本地 CPU 冒烟（0.5B，约 16 分钟）
.venv-data\Scripts\python.exe tools\train\train_sft.py `
  --model models/Qwen2.5-0.5B-Instruct --subset 200 --max-steps 24 `
  --batch-size 1 --grad-accum 8 --max-len 512 --threads 6 --tag smoke

# GPU 全量（7B）—— 见 docs/T3_训练运行手册.md
```

### 4.5 生成 + 评分
```powershell
.venv-data\Scripts\python.exe tools\train\infer_sft.py `
  --model models/Qwen2.5-0.5B-Instruct --adapter models/t3_lora/smoke `
  --data data/processed/official_val.jsonl --limit 20 --threads 6 `
  --out reports/t3_pred_smoke.jsonl

.venv-data\Scripts\python.exe tools\eval\official_scorer.py `
  --pred reports/t3_pred_smoke.jsonl --gold data/processed/official_val.jsonl `
  --allow-partial --out reports/T3_冒烟报告.md
```

---

## 5. 关键结论（含数字，别重复踩）

### 5.1 情绪头管线可用，独立测试仍需重新训练并保留 holdout
| 方案 | Accuracy | Macro-F1 |
|---|---|---|
| 多数类（恒答 anxiety） | 0.3190 | 0.0302 |
| 词典规则层 | 0.2587 | 0.1061 |
| **T2 情绪头（历史完整 val）** | **0.7141** | **0.6221** |

→ 上表统一为历史完整 val 的比较。修正版按会话随机分组（seed=42）校准，holdout Acc 0.7279 / Macro-F1 0.6369，详见最新 `reports/T2_报告.md`。旧检查点曾用完整 val 选优，不能将重新划分的 holdout 当作独立测试集；后续训练已改为仅用同一 calib 选优。
→ 类别不均衡由加权 CE 解决，先验校正温度自动搜出 **tau=0**（不需要额外校正）。

### 5.2 画像头信号不足，建议默认输出空
- 历史工作点 A 的三组 F1：traits 0.227 / interests 0.094 / style 0.287；其全对率 0.0331。
- 历史工作点 B 全对率 0.209，且预测全部为空；同一旧 test 的全空基线也是 0.209，B 未带来增益。0.372 是完整 val 的基线，不能跨子集比较。最新两工作点见 `reports/T2_报告.md`。
- AUC 诊断：平均 **0.59–0.62**（接近随机），`film_animation` 0.42、`casual` 0.45
- **结论：当前一次训练、256 token 输入下，画像头未学到足够信号。** 尚不能判断标注来自哪里，也不能证明长上下文中没有信号。
- 除非官方确认按"分字段 F1"计分，否则**默认输出空画像**收益更高

### 5.3 记忆引用（memory_refs）目前不可预测
- 每个预测点引用 0–9 个 `mem_NNNNNN`，中位 2，仅 2.3% 为空
- ID 是**构建期全局计数器**（每个会话占一段连续区间），train/val 区间重叠
- **公开数据里没有任何地方定义 mem_id 的内容** → 当前缺少可靠的语义检索依据；全空基线 F1 实测 0，不等于证明任意预测方式的期望值严格为 0。

### 5.4 文本生成的真实基线
- 持久化基线（复述上一条助手发言）：ROUGE-L 0.1987 / ROUGE-1 0.3851
- **任何生成模型必须先超过这条线**
- 参考回复中位 129 字、均值 218 字，风格是「高精度情绪镜映 + 不轻易给建议」

---

## 6. 坑与注意事项（都是实际踩过的）

| # | 坑 | 说明 |
|---|---|---|
| 1 | **截断方向** | tokenizer 默认保留前缀、丢弃末尾。本项目最新发言在末尾；T2 保留末尾并保住 `[CLS]`，T3 推理已显式左侧裁剪，与 SFT 的 prompt 裁剪方向一致 |
| 2 | **labels 全屏蔽导致 NaN** | `(p_ids + t_ids)[:max_len]` 在样本超长时会截掉目标回复 → labels 全 -100 → 交叉熵 NaN 并毁掉权重。`train_sft.py` 已改为「先保目标、再从左裁 prompt」 |
| 3 | **镜像下载** | 本机 `raw.githubusercontent.com`、`huggingface.co` 不通；**hf-mirror 的 LFS 文件会 302 到 `cas-bridge.xethub.hf.co`，同样不通**。可用方案：① ModelScope 直连（`tools/train/download_ms.py`，**首选**）② jsdelivr CDN 读 GitHub 上的文本文件（`https://cdn.jsdelivr.net/gh/用户/仓库@分支/路径`）③ hf-mirror 的**小文件**（配置/tokenizer）可用 |
| 4 | **`download.py` 的 Xet** | 用 huggingface_hub 下大文件会走 Xet 协议失败；设 `HF_HUB_DISABLE_XET=1` 也未必成功，建议直接用 ModelScope |
| 5 | **本机无 GPU** | 所有训练都在 24 核 CPU 上跑。T3 的 7B 从未实际训练过，只验证了 0.5B 管线 |
| 6 | **阈值与指标强耦合** | 画像的「分字段 F1」和「三组完全一致率」是**互相冲突**的两个目标（见 `calibrate_t2.py` 的两个工作点）。**必须先确认官方指标再选**，否则白调 |
| 7 | **`app/` 与当前任务不完全匹配** | `app/` 是早期为「自由对话镜像」写的服务层（情绪/记忆/共情/安全/健康检查）。官方评测是**结构化四字段输出**，T5 需要改造 `app/server.py` 的接口契约，`app/emotion.py` 的 7 类词典标签也**无法直接用**（官方是 16 类） |
| 8 | **数据不可外传** | 官方数据集仅限赛事使用，勿公开传播；含真实感心理困扰内容，展示需注意伦理 |
| 9 | 中文路径 | 项目路径含中文，部分命令行工具需注意编码；Python 脚本已用 `encoding='utf-8'` 显式指定 |

---

## 7. 待办与阻塞项

### 7.1 ⛔ 必须先问组委会（阻塞 T2 画像头与 T4）
1. **评测输入是否提供记忆库（mem_id → 内容）？** 不提供则 `memory_refs` 无法预测
2. **四个字段的评分权重与各自指标**：情绪用 Acc 还是 Macro-F1？画像用分字段 F1 还是三组完全一致？
3. **画像标签的标注来源**：能否从对话推断？（当前 AUC 0.6 说明不能）
4. `response_text` 由人评还是模型评？评分维度？
5. 推理时输入是否同样按 `target_user_turn_id` 截断？输出文件格式与提交方式？

### 7.2 T4 记忆引用机制
阻塞于 7.1-①。可解则做检索-选择模块；不可解则做降级（输出空 + 争取其他字段）。

### 7.3 T5 推理服务与镜像交付
- 把四个头组装成**单次推理管线**，输出严格符合官方 schema 的 JSON
- 改造 `app/server.py` 的接口契约（当前是对话式，需改成结构化预测）
- 离线自检、体积/延迟达标、LLM-as-judge + 安全红队自评
- 情绪字段用 T2 分类头**覆盖** LLM 的生成结果（分类头有概率校准，更稳）

### 7.4 可选优化
- 情绪头：2 epoch 或换 `microsoft/mdeberta-v3-base`，预计 Acc 可达 0.75+
- 画像头：若组委会确认可从对话推断，再试「512+ token 上下文 + BCE pos_weight + 拆分独立模型」
- T3：SFT 后加一轮 DPO（0.5–1 卡时），专治"建议给太早"

---

## 8. 许可与合规

| 项 | 说明 |
|---|---|
| 官方数据集 | 仅限赛事使用，**不得外传** |
| `chinese-roberta-wwm-ext` | 见 HF 模型卡（hfl） |
| `Qwen2.5` 系列 | Apache-2.0（**许可最干净，建议坚持用**） |
| 早期调研的 EmoLLM | 仓库 MIT，但**基座各自许可**（InternLM2.5 有单独协议）；若参考其数据请单独核对 |
| 避免 | LLaMA3 系（Meta 社区许可含月活限制条款） |

---

## 9. 时间提醒

赛事报名截止 **2026-10-20**（部分学校校内更早）。距上次更新仅剩十余天，
建议优先级：**问清组委会 > 训 7B > T5 打包 > 画像头优化**。
