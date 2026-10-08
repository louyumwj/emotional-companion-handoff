#!/usr/bin/env bash
# 容器入口：拉起推理后端 -> 等就绪 -> 预热 -> 启动业务 API
# 设计原则：任何一步失败都不能让容器挂掉，必须降级到规则模式继续提供服务。
set -uo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/models/qwen2.5-7b-awq}"
LLM_PORT="${LLM_PORT:-23333}"
PORT="${PORT:-8000}"
mkdir -p /workspace/logs /workspace/state

log() { echo "[entrypoint] $(date '+%H:%M:%S') $*"; }

# ---------- 1. 推理后端 ----------
if [[ "${LLM_BACKEND:-openai}" == "openai" ]]; then
    if ! command -v lmdeploy >/dev/null 2>&1; then
        log "!! lmdeploy 未安装 -> 降级 LLM_BACKEND=none"
        export LLM_BACKEND=none
    elif [[ ! -d "$MODEL_DIR" ]]; then
        log "!! 模型目录不存在: $MODEL_DIR -> 降级 LLM_BACKEND=none"
        export LLM_BACKEND=none
    else
        log "启动 LMDeploy api_server: $MODEL_DIR (format=${MODEL_FORMAT:-awq}, tp=${TP:-1})"
        lmdeploy serve api_server "$MODEL_DIR" \
            --server-name 127.0.0.1 \
            --server-port "$LLM_PORT" \
            --model-format "${MODEL_FORMAT:-awq}" \
            --cache-max-entry-count "${CACHE_MAX_ENTRY_COUNT:-0.4}" \
            --tp "${TP:-1}" \
            > /workspace/logs/lmdeploy.log 2>&1 &
        LLM_PID=$!

        ready=0
        for i in $(seq 1 180); do
            if curl -fsS "http://127.0.0.1:${LLM_PORT}/v1/models" >/dev/null 2>&1; then
                log "推理后端就绪（${i}s）"
                ready=1
                break
            fi
            if ! kill -0 "$LLM_PID" 2>/dev/null; then
                log "!! 推理后端进程退出，见 /workspace/logs/lmdeploy.log -> 降级"
                export LLM_BACKEND=none
                break
            fi
            sleep 1
        done
        if [[ "$ready" == "0" && "${LLM_BACKEND}" == "openai" ]]; then
            log "!! 等待超时 -> 降级 LLM_BACKEND=none"
            export LLM_BACKEND=none
        fi
    fi
fi
log "LLM_BACKEND=${LLM_BACKEND}"

# ---------- 2. 业务 API ----------
python3 -m uvicorn app.server:app --host 0.0.0.0 --port "$PORT" --workers 1 \
    > /workspace/logs/app.log 2>&1 &
APP_PID=$!

cleanup() { log "收到退出信号，关闭子进程"; kill -TERM "$APP_PID" 2>/dev/null; [[ -n "${LLM_PID:-}" ]] && kill -TERM "$LLM_PID" 2>/dev/null; }
trap cleanup TERM INT

for i in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        log "业务 API 就绪（${i}s）"
        break
    fi
    sleep 1
done

# ---------- 3. 预热 ----------
# 目的：把首次推理的编译/显存分配开销提前吃掉，避免评测第一个请求超时。
if [[ "${LLM_BACKEND}" == "openai" ]]; then
    log "预热首轮推理..."
    curl -fsS -m 300 -X POST "http://127.0.0.1:${PORT}/chat" \
        -H 'Content-Type: application/json' \
        -d '{"session_id":"warmup","user_id":"warmup","message":"你好"}' \
        > /workspace/logs/warmup.json 2>&1 \
        && log "预热完成" || log "!! 预热失败（不阻塞启动）"
fi

log "服务已就绪：http://0.0.0.0:${PORT}"
wait "$APP_PID"
