# 一键环境复原（Windows / PowerShell）
#
#   pwsh -File setup.ps1              # 完整安装
#   pwsh -File setup.ps1 -SkipModels  # 跳过基座模型下载
#
# 做三件事：
#   1. 建虚拟环境 .venv-data
#   2. 安装依赖（默认走清华源，国内快）
#   3. 从 ModelScope 下载两个基座模型（hf-mirror 的 LFS 在本机不通）
#
# 不含在包内的东西（约 2.2 GB，本脚本负责再生）：
#   .venv-data/                        依赖环境
#   models/chinese-roberta-wwm-ext/    T2 编码器（393 MB）
#   models/Qwen2.5-0.5B-Instruct/      T3 冒烟用（953 MB）
# 注意：models/t2_multitask/best.pt（已训练权重）在完整包里已包含，无需重训。

param(
    [switch]$SkipModels,
    [string]$Python = "python",
    [string]$PipIndex = "https://pypi.tuna.tsinghua.edu.cn/simple"
)

$ErrorActionPreference = "Stop"
function Assert-NativeSuccess([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE)" }
}
$ws = $PSScriptRoot
Set-Location $ws

Write-Host "=== 1/3 创建虚拟环境 ===" -ForegroundColor Cyan
if (-not (Test-Path "$ws\.venv-data\Scripts\python.exe")) {
    & $Python -m venv "$ws\.venv-data"
    Assert-NativeSuccess "Create virtual environment"
    Write-Host "  已创建 .venv-data"
} else {
    Write-Host "  .venv-data 已存在，跳过"
}
$py = "$ws\.venv-data\Scripts\python.exe"

Write-Host "`n=== 2/3 安装依赖（清华源）===" -ForegroundColor Cyan
$env:PIP_INDEX_URL = $PipIndex
& $py -m pip install -q --upgrade pip
Assert-NativeSuccess "Upgrade pip"
& $py -m pip install -q -r "$ws\requirements.txt" -i $PipIndex
Assert-NativeSuccess "Install service dependencies"
& $py -m pip install -q torch transformers peft accelerate numpy sentencepiece `
    opencc-python-reimplemented -i $PipIndex
Assert-NativeSuccess "Install training dependencies"
Write-Host "  依赖安装完成"

Write-Host "`n=== 3/3 下载基座模型（ModelScope）===" -ForegroundColor Cyan
$env:HTTPS_PROXY = ""; $env:HTTP_PROXY = ""
if ($SkipModels) {
    Write-Host "  已跳过（-SkipModels）"
} else {
    & $py "$ws\tools\train\download_ms.py" "dienstag/chinese-roberta-wwm-ext" `
        "$ws\models\chinese-roberta-wwm-ext" `
        "config.json,vocab.txt,pytorch_model.bin" `
        --optional-files "tokenizer.json,tokenizer_config.json,special_tokens_map.json"
    Assert-NativeSuccess "Download RoBERTa"

    & $py "$ws\tools\train\download_ms.py" "Qwen/Qwen2.5-0.5B-Instruct" `
        "$ws\models\Qwen2.5-0.5B-Instruct" `
        "config.json,generation_config.json,merges.txt,tokenizer.json,tokenizer_config.json,vocab.json,model.safetensors"
    Assert-NativeSuccess "Download Qwen"
}

# roberta 的 tokenizer_config.json 若缺失会让 AutoTokenizer 走慢速分词器
$tc = "$ws\models\chinese-roberta-wwm-ext\tokenizer_config.json"
if ((Test-Path "$ws\models\chinese-roberta-wwm-ext") -and -not (Test-Path $tc)) {
    [System.IO.File]::WriteAllText($tc, '{"model_max_length": 512, "do_lower_case": false}', [System.Text.UTF8Encoding]::new($false))
    Write-Host "  已补写 tokenizer_config.json"
}

Write-Host "`n=== 自检 ===" -ForegroundColor Cyan
if ($SkipModels) {
    & $py "$ws\tools\train\verify_models.py" --skip-models
} else {
    & $py "$ws\tools\train\verify_models.py"
}
Assert-NativeSuccess "Verify environment"

Write-Host "`n环境就绪。下一步：" -ForegroundColor Green
Write-Host "  1) 读 HANDOFF.md 和 reports/T2T3_完成报告.md"
Write-Host "  2) 重建数据: .venv-data\Scripts\python.exe tools\data\build_official_dataset.py"
Write-Host "  3) 复现 T2:  .venv-data\Scripts\python.exe tools\train\calibrate_t2.py --threads 20"
Write-Host "  4) 训 7B:    见 docs\T3_训练运行手册.md（需 GPU）"
