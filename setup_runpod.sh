#!/bin/bash
# Setup script for RunPod — installs deps for offline/sample_rollouts.py.
#
# Blackwell pods often ship with a CUDA 12.x system toolkit while current vLLM
# wheels pull Torch CUDA 13.x. Build CUDA extensions against the pip CUDA 13
# toolchain below, not /usr/local/cuda, or mamba_ssm will fail with a CUDA
# mismatch.
#
# Usage: bash setup_runpod.sh [HF_TOKEN]

set -e

HF_TOKEN="${1:-$HF_TOKEN}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$WORKSPACE_DIR/hf-cache}"
TMP_WORK_DIR="${TMP_WORK_DIR:-$WORKSPACE_DIR/tmp}"

echo "=== System deps ==="
apt-get update -qq && apt-get install -y -qq \
    git \
    build-essential \
    clang \
    ninja-build \
    curl \
    ca-certificates

echo "=== Workspace cache dirs ==="
mkdir -p "$HF_CACHE_DIR" "$TMP_WORK_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export HF_HUB_CACHE="$HF_CACHE_DIR/hub"
export HF_XET_CACHE="$HF_CACHE_DIR/xet"
export TMPDIR="$TMP_WORK_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "=== Detect GPU arch ==="
GPU_ARCH=$(python3 -c "
import subprocess
r = subprocess.run(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],
                  capture_output=True, text=True)
cap = r.stdout.strip().split('\n')[0].replace('.','')
print(cap)
")
echo "GPU compute capability: $GPU_ARCH"
TORCH_ARCH="${GPU_ARCH:0:2}.${GPU_ARCH:2}"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_ARCH"

echo "=== Python packaging base ==="
pip install -U pip setuptools wheel packaging ninja

echo "=== vLLM + Torch ==="
# Do not pre-pin torch here. On Blackwell, current vLLM pulls a CUDA 13 Torch
# stack; pinning an older cu12 Torch causes libcudart/CUDA mismatch failures.
pip uninstall -y torchaudio >/dev/null 2>&1 || true
pip install -U vllm

echo "=== transformers + peft + deps ==="
pip install -U \
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

echo "=== CUDA 13 runtime/toolchain for vLLM + extension builds ==="
# The old *-cu13 package names are deprecated stubs. These package names provide
# libcudart/libnvrtc/cublas and nvcc that match Torch CUDA 13.x.
pip install -U \
    nvidia-cuda-runtime \
    nvidia-cuda-nvrtc \
    nvidia-cuda-nvcc \
    nvidia-cublas \
    "nvidia-cudnn-cu13==9.19.0.56"

CUDA_HOME=$(python3 - <<'PY'
import glob
import os
import site

roots = site.getsitepackages() + [site.getusersitepackages()]
for root in roots:
    for path in glob.glob(os.path.join(root, "nvidia", "cuda_nvcc")):
        if os.path.exists(os.path.join(path, "bin", "nvcc")):
            print(path)
            raise SystemExit
raise SystemExit("Could not find pip-installed nvidia/cuda_nvcc/bin/nvcc")
PY
)
NVIDIA_LIB_PATHS=$(python3 - <<'PY'
import glob
import os
import site

paths = []
for root in site.getsitepackages() + [site.getusersitepackages()]:
    paths.extend(glob.glob(os.path.join(root, "nvidia", "*", "lib")))
    paths.extend(glob.glob(os.path.join(root, "nvidia", "*", "lib64")))
print(":".join(dict.fromkeys(paths)))
PY
)
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$NVIDIA_LIB_PATHS:${LD_LIBRARY_PATH:-}"
echo "CUDA_HOME: $CUDA_HOME"

echo "=== Torch/CUDA sanity ==="
python3 - <<'PY'
import os
import subprocess

import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
subprocess.run([os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc"), "--version"], check=True)
PY

echo "=== mamba_ssm + causal_conv1d (required for Nemotron-H) ==="
# Needed for Nemotron-H custom model code. Build against the CUDA_HOME above.
# This can take 10-30 minutes on a fresh RunPod image.
MAX_JOBS="${MAX_JOBS:-4}" \
TORCH_CUDA_ARCH_LIST="$TORCH_ARCH" \
CUDA_HOME="$CUDA_HOME" \
PATH="$CUDA_HOME/bin:$PATH" \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    pip install -U mamba_ssm==2.3.1 causal_conv1d==1.6.1 --no-build-isolation

echo "=== HuggingFace login ==="
if [ -n "$HF_TOKEN" ]; then
    hf auth login --token "$HF_TOKEN"
else
    echo "WARNING: No HF_TOKEN provided. Set HF_TOKEN env var or pass as arg if model is gated."
fi

echo "=== Persist RunPod env ==="
cat > "$WORKSPACE_DIR/nemotron_runpod_env.sh" <<EOF
export HF_HOME="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_HUB_CACHE="$HF_HUB_CACHE"
export HF_XET_CACHE="$HF_XET_CACHE"
export TMPDIR="$TMPDIR"
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="$NVIDIA_LIB_PATHS:\${LD_LIBRARY_PATH:-}"
EOF
if ! grep -q "nemotron_runpod_env.sh" ~/.bashrc 2>/dev/null; then
    echo "source $WORKSPACE_DIR/nemotron_runpod_env.sh" >> ~/.bashrc
fi

echo ""
echo "=== Setup complete! ==="
echo "Run now: source $WORKSPACE_DIR/nemotron_runpod_env.sh"
echo "Smoke: python offline/sample_rollouts.py --mode probe --model_path unsloth/Nemotron-3-Nano-30B-A3B --adapter_path /workspace/nemotron-reasoning-lora-adapter --output /workspace/rollouts_smoke.jsonl --group_size 1 --max_problems 1 --batch_size 1"
