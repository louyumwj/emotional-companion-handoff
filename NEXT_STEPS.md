# 情感陪伴镜像：后续开发交接

更新日期：2026-10-08。本文记录修正版的实际状态和下一阶段任务；历史文档中的预算、目标和未验证推测不能当作已完成结果。

## 1. 交付入口

- 私有仓库：<https://github.com/louyumwj/emotional-companion-handoff>
- 修正版 Release：<https://github.com/louyumwj/emotional-companion-handoff/releases/tag/handoff-20261008>
- 代码和本文在仓库中；完整修正版 ZIP、本文、SHA256 校验文件在 Release 附件中。
- ZIP：`情感陪伴镜像_交接包_修正版_20261008.zip`，438,017,049 字节。
- ZIP SHA256：`101ac1e54c5675a4ca92cc8bfe317c833f166b769a280591089ce089b31f67ea`。

Git 源码不存放原始 JSONL、逐条预测、样例对话和二进制模型权重；这些文件保留在私有 Release ZIP 中。`SOURCE_IMPORT.json` 列出源码导入记录及需要恢复的文件。`ARCHIVE_MANIFEST_SHA256.json` 对应 ZIP 原始内容，不用于校验后来新增或修改的仓库文档。

## 2. 当前状态

目标是给每个预测点输出 `response_text`、`emotion_label`、`user_profile`、`memory_refs`。现有 `app/` 仍是自由对话服务骨架，尚未完成官方四字段在线接口。

| 模块 | 已完成 | 后续工作 |
| --- | --- | --- |
| T1 数据 | 解析、按目标用户轮截断、生成训练视图 | 在接手环境重建，核对数量和泄漏 |
| T2 情绪/画像 | 已训练旧检查点，修复计分、路径、校准分组 | 重新训练保留独立 holdout；按官方画像指标选工作点 |
| T3 回复 | 0.5B LoRA 冒烟，截断与条件注入修复 | GPU 验证、7B 正式 SFT、同子集比较与全量评测 |
| T4 记忆引用 | 已分析公开数据 | 等待记忆库/接口定义，开发检索或记录降级方案 |
| T5 服务/镜像 | 基础 API、会话隔离、危机兜底 | 四字段管线、正式契约、离线镜像与性能验收 |

修复详情见 `FIXES_20261008.md`。本次没有重新训练模型：T2 的 207 个张量值未变，T3 adapter 未变。T2 校准文件与预测报告已经更新。

已完成 10 项回归检查、全量 T2 校准/指标复算、T3 adapter 加载及单样本 8-token 生成，以及 ZIP CRC 和内容哈希校验。验证记录见 `reports/修正版验证结果.json`。

**未完成验证：**GPU 执行、7B 下载/训练、Docker 构建、全新机器联网安装。完整 ZIP 仍不含基座模型、虚拟环境和可再生训练视图。

## 3. 接手和恢复

需要有权访问该私有仓库的 GitHub 账号。下载后先验 SHA256，再解压到新的项目目录；进入 ZIP 内的 `情感陪伴镜像_完整` 目录。源码仓库可用于跟踪后续代码修改，完整 ZIP 可用于恢复训练数据和权重。

```powershell
gh release download handoff-20261008 --repo louyumwj/emotional-companion-handoff --pattern '*修正版*.zip' --dir .\download
Get-FileHash -LiteralPath '.\download\情感陪伴镜像_交接包_修正版_20261008.zip' -Algorithm SHA256
Expand-Archive -LiteralPath '.\download\情感陪伴镜像_交接包_修正版_20261008.zip' -DestinationPath .\restored
Set-Location '.\restored\情感陪伴镜像_完整'
pwsh -File setup.ps1
.venv-data\Scripts\python.exe tools\data\build_official_dataset.py
.venv-data\Scripts\python.exe tools\train\verify_models.py
.venv-data\Scripts\python.exe -m unittest discover -s tests -v
```

验收：train 18,653 个预测点，val 2,354 个预测点；模型自检通过；测试输出应为 10 项通过且无跳过。`-SkipModels` 只验证依赖，不代表模型可以推理。

若从 Git clone 继续开发：将 ZIP 内 `SOURCE_IMPORT.json` 的 `restore_from_release` 列表对应文件复制到 clone 的相同相对路径，再执行上述安装和重建命令。保留仓库最新版代码；这些恢复文件已由 `.gitignore` 排除。

## 4. 优先级与验收

### P0：确认正式评测契约

负责人：项目负责人向组委会确认，开发者记录答案。尚未代替用户联系外部人员。

1. 输入是否提供 `mem_id -> 内容` 的记忆库，或其他生成/查询记忆 ID 的规则？
2. 四字段指标与权重，尤其画像是分字段 F1 还是三组完全一致率？
3. 情绪指标、画像标注依据、回复评审维度和适用语言？
4. 输入是否已经截断到 `target_user_turn_id`？批量/HTTP 格式、鉴权和重置语义是什么？
5. GPU、镜像体积、冷启动、超时、并发、离线运行和提交限制是什么？

验收：把答复与来源写入 `docs/待确认清单.md`；新增正式请求/响应样例。记忆库缺失说明目前缺少可靠的语义检索依据，不能证明任何预测方式都不可能有效。当前画像 AUC 也不能证明对话中没有画像信号。

### P1：恢复环境，冻结数据与评测

负责人：接手开发者。先执行第 3 节，记录 Python、PyTorch、Transformers、PEFT 版本和设备信息。

- 使用 `tools/train/labels.py` 的官方标签；服务层 7 类词典标签不能直接充当官方 16 类输出。
- 输入只保留目标用户轮及之前的历史，不包含目标助手答案或未来轮次。
- 会话分组随机种子固定为 42；train/calibrate 的 `--split-seed` 保持一致。
- 保存模型、数据和依赖版本以及运行参数，新增实验使用独立目录。

验收：数量一致、无会话跨 calib/holdout、无未来文本泄漏、10 项测试通过。

### P2：T2 重新训练和独立评估

负责人：模型开发者。旧检查点曾使用完整 val 选优，当前分组 holdout 只能作为诊断分数。后续只在 calib 上选检查点、阈值与超参数，最终固定方案再看 holdout。

先建立独立实验副本并进入该目录，再执行下列命令。当前校准脚本的输出目录固定为 `models/t2_multitask/` 和 `reports/`，没有 `--out` 参数；不能在唯一的发布基线目录中直接重跑。

```powershell
.venv-data\Scripts\python.exe tools\train\train_multitask.py --epochs 1 --batch-size 16 --max-length 256 --freeze-layers 6 --lr 3e-5 --threads 12 --split-seed 42 --tag retrain --out models/t2_multitask
.venv-data\Scripts\python.exe tools\train\calibrate_t2.py --ckpt models/t2_multitask/best.pt --split-seed 42 --threads 12
```

运行前核对脚本 `--help`。画像增加长上下文、类别权重或独立模型属于待试验方案，不保证分数提升。先确定官方指标，再选择 A/B 工作点或空画像基线。

现有诊断结果：calib 1,178 / holdout 1,176；holdout 情绪 Acc 0.7279、Macro-F1 0.6369。画像 A 全对率 0.0680，B 为 0.3673，同子集空画像基线约 0.3682。工作点和基线必须按同一组实例比较。

验收：新报告明确数据分组、完整预测覆盖率、情绪指标、三组画像指标和全对率，并注明哪些数据参与了调参。不得反复按 holdout 结果选模型。

### P3：T3 GPU 冒烟、正式训练和评估

负责人：有 GPU 的模型开发者。以 `docs/T3_训练运行手册.md` 为命令参考；其中旧卡时、显存和性能目标是估算，先实测再排期。

1. 按手册下载 7B 的索引及四个分片，运行模型自检。
2. 先用 10-20 步、较小 batch、`--gradient-checkpointing` 做 GPU 冒烟，记录峰值显存、步时与 loss，检查 NaN/OOM。
3. 再决定 batch、累积步数、上下文长度和训练时长；保存独立 adapter、参数与日志。
4. 基座与 adapter 在相同实例、相同生成参数下比较；不拿历史 20 条 smoke 分数与 2,354 条基线比较。
5. 选定方案后生成完整 val 文件并用默认严格覆盖率评分。

```powershell
.venv-data\Scripts\python.exe tools\train\infer_sft.py --model models/Qwen2.5-7B-Instruct --adapter models/t3_lora/v1 --data data/processed/official_val.jsonl --limit 0 --t2-pred reports/t2_pred_val.jsonl --out reports/t3_pred_val.jsonl
.venv-data\Scripts\python.exe tools\eval\official_scorer.py --pred reports/t3_pred_val.jsonl --gold data/processed/official_val.jsonl --out reports/T3_正式报告.md
```

`--limit 0` 表示全量。局部冒烟需显式 `--allow-partial` 并报告覆盖率。评分器会拒绝重复 ID、未知 ID 和缺失预测。启用 T2 注入时，缺少所需实例预测会报错。

验收：GPU 日志可复现、全量 2,354 条覆盖、同子集基线比较，以及共情、过早建议、重复、诊断/开药、危机场景的人工抽查。DPO、合并和量化放在 SFT 基线验证之后。

### P4：T4 记忆引用

依赖 P0。有正式记忆库时实现候选检索、选择和合法 ID 校验，保存同子集 precision/recall/F1。没有可用库时明确记录 `memory_refs=[]` 的临时降级及指标损失，不把数字 ID 顺序猜测当作已验证语义检索。

验收：返回 ID 可追溯、无跨用户引用；重置会话清理关联记忆；样例和评测报告可复现。

### P5：T5 四字段服务与镜像

负责人：服务开发者。先实现独立的单次预测管线，再接入 `app/server.py` 的正式接口。保持离线批量和在线接口使用同一推理逻辑。

- 输出严格包含官方四字段，使用官方标签；T2 提供情绪/画像，T3 提供回复，T4 提供记忆引用。
- 匿名请求无状态；明确用户/会话时才保存历史；外部 system 历史过滤；显式空历史与 reset 语义保留。
- 模型缺失、后端超时、坏输入和长输入的返回行为按正式契约定义；健康检查区分存活与模型就绪。
- 权重和依赖在运行前准备好，验证断网运行、冷启动、镜像大小、并发及 P50/P95 延迟。
- 危机层目前是规则兜底，测试不能证明全面覆盖；按实际部署场景继续检查误报和漏报，转介信息需核对。

验收：Docker 真机构建和 GPU 启动通过；正式样例/批量预测、会话隔离、reset、异常和危机用例通过；在断网环境完成端到端评测。性能门槛采用 P0 确认值，不沿用历史猜测。

## 5. 每次交接应附的记录

提交实验参数、依赖和设备版本、数据/模型 SHA256、预测覆盖率、同子集比较报告、耗时与显存，以及仍未验证的部分。新权重和原始数据使用私有 Release/受控存储，不直接提交 Git。官方数据仅限赛事用途，仓库保持私有；对外分享前先处理数据与许可边界。

阅读顺序：本文 -> `FIXES_20261008.md` -> `reports/修正版验证结果.json` -> 最新 `reports/T2_报告.md` -> 对应模块代码。原 `HANDOFF.md`、`reports/T2T3_完成报告.md` 和服务 README 作为历史参考，不代表 T4/T5 或 7B 正式训练已经完成。
