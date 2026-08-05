#!/bin/bash
set -e

########################################
# Config
########################################

#MODEL_DIR=/data/kcl/myt/ROLL/outputs/envscaler_non_conv_rl_20260518_005957/checkpoint/20260518-005957/checkpoint-10/hf_converted
#MODEL_DIR=/data/kcl/myt/ROLL/outputs/envscaler_non_conv_rl_20260517_020626/checkpoint/20260517-020626/checkpoint-10/hf_converted
#MODEL_DIR=/data/kcl/myt/saves/sft_Qwen3-8B
#MODEL_DIR=/data/kcl/myt/Qwen3-4B
#MODEL_DIR=/data/kcl/myt/saves/sft_Qwen3-4B
MODEL_DIR=/data/kcl/myt/ROLL/outputs/envscaler_non_conv_rl_20260510_165403/checkpoint/20260510-165403/checkpoint-10/hf_converted
#MODEL_DIR=/data/kcl/myt/ROLL/outputs/envscaler_non_conv_rl_20260519_070939/checkpoint/20260519-070939/checkpoint-10/hf_converted

# SERVED_NAME=qwen3-8b-sft_rl
# SERVED_NAME=qwen3-8b-sft_secrl
# SERVED_NAME=qwen3-8b-sft
#SERVED_NAME=qwen3-4b-sft_secrl
SERVED_NAME=qwen3-4b-sft_rl
# SERVED_NAME=qwen3-4b-sft

PORT=8315
TP_SIZE=1
MAX_LEN=32678
GPU_UTIL=0.6                  # 降低显存占用
DTYPE=float16
MAX_NUM_SEQS=1024               # warm-up 用的 dummy requests，减少 OOM 风险

########################################
# Select lowest memory GPU
########################################

echo "Checking GPU usage..."

GPU_INFO=$(nvidia-smi \
    --query-gpu=index,memory.used \
    --format=csv,noheader,nounits)

echo "GPU usage (index, memory_used_MB):"
echo "$GPU_INFO"

GPU_LIST=$(echo "$GPU_INFO" \
    | sort -t',' -k2 -n \
    | head -n "$TP_SIZE" \
    | cut -d',' -f1 \
    | tr '\n' ',' \
    | sed 's/,$//')

echo ""
echo "Selected GPUs: $GPU_LIST"
echo ""

export CUDA_VISIBLE_DEVICES=$GPU_LIST

########################################
# PyTorch CUDA config
########################################

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export VLLM_MAX_NUM_SEQS=$MAX_NUM_SEQS
export VLLM_ATTENTION_BACKEND=xformers

########################################
# Start vLLM server
########################################

echo "Starting vLLM server..."
echo "Model path: $MODEL_DIR"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Max Model Length: $MAX_LEN"
echo "GPU Utilization: $GPU_UTIL"
echo "Max warm-up sequences: $MAX_NUM_SEQS"
echo ""

python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_DIR \
  --tensor-parallel-size $TP_SIZE \
  --dtype $DTYPE \
  --max-model-len $MAX_LEN \
  --gpu-memory-utilization $GPU_UTIL \
  --served-model-name $SERVED_NAME \
  --port $PORT \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --no-enable-prefix-caching \
  --disable-custom-all-reduce \
  --uvicorn-log-level warning