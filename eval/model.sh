#!/usr/bin/env bash
# 在 interact_with_env 目录下执行: ./model.sh
# 主要改下面 export 块里的参数即可。

set -euo pipefail

export MODEL_DIR=/home/mouyutao/Models/Qwen3-8B
export TP_SIZE=1                    # 单卡填 1；多卡再加大
export DTYPE=bfloat16               # 与 config 一致
export MAX_LEN=32768                # 显存紧可改 4096；有余量可试 16384
export GPU_UTIL=0.90                # 占满 GPU 时可略降到 0.85~0.9
export SERVED_NAME=Qwen3-8B         # 客户端里写的 model 名字
export PORT=8012
export HOST=0.0.0.0                 # 仅本机访问可改为 127.0.0.1

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --dtype auto \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --served-model-name "${SERVED_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code
