# 2026 动感地带AI+高校创智计划 · AI技术赛道
# 赛题二：数字人综合情感陪伴对话模型
# 交付物：可被评测平台直接拉起并调用 HTTP 接口的镜像
#
# 本地构建：  docker build -t emotional-companion:0.1 .
# 本地运行：  docker run --gpus all -p 8000:8000 -v "%cd%/models:/workspace/models" emotional-companion:0.1
# 无 GPU：    docker run -p 8000:8000 -e LLM_BACKEND=none emotional-companion:0.1   （规则兜底模式）

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ⚠️ 务必换成咪咕仝学提供的「赛事基础镜像」：
#    它自带 CUDA / PyTorch / 推理框架，能省掉几 GB 体积和大量构建时间。
# FROM registry.xxx/base:cuda12.4-py310-torch2.4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=8000 \
    LLM_PORT=23333 \
    LLM_BACKEND=openai \
    LLM_BASE_URL=http://127.0.0.1:23333/v1 \
    LLM_MODEL=qwen2.5-7b-awq \
    MODEL_DIR=/workspace/models/qwen2.5-7b-awq \
    EMBED_MODEL_DIR=/workspace/models/bge-small-zh-v1.5 \
    MODEL_FORMAT=awq \
    STATE_DIR=/workspace/state

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN mkdir -p /workspace/logs /workspace/state

COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# 可选依赖（sentence-transformers / faiss / lmdeploy）；按需取消注释
# COPY requirements-optional.txt /workspace/requirements-optional.txt
# RUN pip3 install --no-cache-dir -r /workspace/requirements-optional.txt

COPY app/ /workspace/app/
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# 权重是否打进镜像，取决于私有镜像配额：
#   A) 打进镜像（评测最稳，体积大）→ 取消下面两行注释
# COPY models/ /workspace/models/
# B) 不进镜像 → 运行时 -v 挂载到 /workspace/models，评测平台大概率不允许，
#    因此正式提交建议走 A 或让平台把权重放在镜像里。
RUN mkdir -p /workspace/models

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=20 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["/workspace/entrypoint.sh"]
