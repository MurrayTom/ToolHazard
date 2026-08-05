#!/bin/bash
set -e

########################################
# Config
########################################

MODEL_DIR=/home/mouyutao/yinzhe/EnvScaler/scen_generator/data/models/Qwen3Guard-Gen-8B
PORT=8000
TP_SIZE=1
MAX_LEN=8192
GPU_UTIL=0.6                  # 降低显存占用
DTYPE=float16
SERVED_NAME=qwen-guard
MAX_NUM_SEQS=1024               # warm-up 用的 dummy requests，减少 OOM 风险

########################################
# Select lowest memory GPU
########################################

echo "Checking GPU usage..."
GPU_INFO=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

echo "GPU usage (index, memory_used_MB):"
echo "$GPU_INFO"

BEST_GPU=$(echo "$GPU_INFO" | sort -t',' -k2 -n | head -n1 | cut -d',' -f1)

echo ""
echo "Selected GPU: $BEST_GPU"
echo ""

export CUDA_VISIBLE_DEVICES=$BEST_GPU

########################################
# PyTorch CUDA config
########################################

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export VLLM_MAX_NUM_SEQS=$MAX_NUM_SEQS

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
  --port $PORT
