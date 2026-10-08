# 情感陪伴镜像

修正版源码与下一阶段开发交接，更新于 2026-10-08。项目根据对话历史输出 `response_text`、`emotion_label`、`user_profile`、`memory_refs` 四个字段。

接手先读 [后续任务交接](NEXT_STEPS.md)，再读 [修复记录](FIXES_20261008.md) 和 [验证结果](reports/修正版验证结果.json)。

## 下载完整修正版

[Release handoff-20261008](https://github.com/louyumwj/emotional-companion-handoff/releases/tag/handoff-20261008) 包含完整修正版 ZIP、最新交接文档与 SHA256 校验文件。需要有权访问本私有仓库的 GitHub 账号。

```powershell
gh release download handoff-20261008 --repo louyumwj/emotional-companion-handoff --pattern 'emotional-companion-fixed-20261008.zip' --dir .\download
Get-FileHash -LiteralPath '.\download\emotional-companion-fixed-20261008.zip' -Algorithm SHA256
```

GitHub 使用英文附件名 `emotional-companion-fixed-20261008.zip`；本地中文名称的 ZIP 与它内容相同。预期 SHA256：`101ac1e54c5675a4ca92cc8bfe317c833f166b769a280591089ce089b31f67ea`。

Git 跟踪代码、schema、配置和报告；完整数据、逐条预测与二进制模型权重放在 Release。恢复文件清单见 [SOURCE_IMPORT.json](SOURCE_IMPORT.json)。ZIP 内不含基座模型、虚拟环境和可再生训练视图，恢复步骤见交接文档。

## 当前状态

| 模块 | 状态 |
| --- | --- |
| 数据处理 | 已实现目标轮截断与实例/训练视图生成 |
| T2 情绪/画像 | 旧检查点可用，计分/校准修复；独立 holdout 需重训 |
| T3 回复生成 | 0.5B LoRA 冒烟通过，7B 正式训练待 GPU |
| T4 记忆引用 | 等待官方记忆库或 ID 规则确认 |
| T5 服务/镜像 | 对话服务骨架可用，正式四字段接口及 Docker 待完成 |

10 项回归检查、全量 T2 校准与指标复算、T3 adapter 单样本加载/生成、ZIP CRC 和内容 SHA256 已通过。未执行 GPU、7B 下载/训练、Docker 构建或全新机器联网安装。本次未重训模型。

## 目录

- `app/`：服务、记忆、情绪规则、共情和安全模块。
- `tools/data/`、`tools/train/`、`tools/eval/`：数据处理、训练/推理与评分脚本。
- `tests/`：针对本次修复的回归检查。
- `docs/`：待确认事项、T3 GPU 训练参考和 [历史服务说明](docs/SERVICE_README.md)。
- `reports/`：评测报告和验证记录；历史分数保留，不得与不同子集直接比较。
- `models/`：校准与 adapter 配置；二进制权重从完整 ZIP 恢复。

[原 HANDOFF](HANDOFF.md) 保留历史背景；下一阶段任务和状态以 `NEXT_STEPS.md` 为准。`ARCHIVE_MANIFEST_SHA256.json` 对应发布 ZIP 内的内容，`SOURCE_IMPORT.json` 记录最初导入，不用于验证后续代码修改。

官方数据限赛事授权使用，本仓库保持私有。
