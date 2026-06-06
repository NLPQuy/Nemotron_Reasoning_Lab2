#!/bin/bash
# Setup script for RunPod — installs all deps to run generate_rollouts_vllm.py
# Tested on: Ubuntu 22.04 + CUDA 12.x devel image
# Usage: bash setup_runpod.sh [HF_TOKEN]

set -e

HF_TOKEN="${1:-$HF_TOKEN}"

echo "=== System deps ==="
apt-get update -qq && apt-get install -y -qq git build-essential clang

echo "=== Detect GPU arch ==="
GPU_ARCH=$(python3 -c "
import subprocess
r = subprocess.run(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],
                  capture_output=True, text=True)
cap = r.stdout.strip().split('\n')[0].replace('.','')
print(cap)
")
echo "GPU compute capability: $GPU_ARCH"

echo "=== PyTorch ==="
# vllm requires torch 2.4+ — use 2.6 + cu124 (widely compatible)
pip install -q torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124

echo "=== vLLM ==="
pip install -q vllm

echo "=== transformers + peft + deps ==="
pip install -q \
	    "transformers>=4.56.2" \
	    "peft>=0.15.0" \
	    "accelerate>=1.0.0" \
	    "huggingface_hub>=0.36.2" \
	    "hf-transfer>=0.1.9" \
	    "safetensors>=0.5.0" \
	    "datasets" \
	    "sentencepiece" \
	    "numpy" \
	    "llmlingua" \
	    "kaggle"

echo "=== mamba_ssm + causal_conv1d (build from source) ==="
# Needed for Nemotron-H custom model code (trust_remote_code=True)
# This takes ~10-15 min
TORCH_CUDA_ARCH_LIST="${GPU_ARCH:0:2}.${GPU_ARCH:2}" \
    pip install -q mamba_ssm==2.3.1 causal_conv1d==1.6.1 --no-build-isolation

echo "=== HuggingFace login ==="
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
    echo "WARNING: No HF_TOKEN provided. Set HF_TOKEN env var or pass as arg if model is gated."
fi

echo "=== Enable hf-transfer for faster downloads ==="
export HF_HUB_ENABLE_HF_TRANSFER=1

echo ""
echo "=== Setup complete! ==="
echo "Run: python generate_rollouts_vllm.py --help"
