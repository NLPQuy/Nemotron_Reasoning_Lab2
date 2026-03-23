# Workflow Report — NVIDIA Nemotron Reasoning Challenge (Kaggle)

> **Competition:** nvidia-nemotron-3-reasoning-challenge
> **Model:** Nemotron-3-Nano-30B-A3B (BF16, Hybrid Mamba+Attention MoE)
> **Task:** Fine-tune LoRA adapter (max rank 32), submit `submission.zip`
> **GPU:** Kaggle RTX PRO 6000 (Blackwell architecture) — **Internet DISABLED**

---

## 1. Tổng quan 4 File

| File | Approach | LoRA Target | Dataset Size |
|---|---|---|---|
| `nvidia_nemotron_submission_demo.py` | Demo baseline, không có training thực | `in_proj, out_proj, up_proj, down_proj` | Full train |
| `nemotron_unsloth_sft_training.py` | SFT với Unsloth (4-bit NF4), `\boxed{}` only | All major projections | Full train |
| `nvidia_nemotron_sft.py` | SFT + Dynamic CoT (`<think>` tags), `all-linear` | All linear layers | 6,000 samples |
| `nvidia_nemotron_tuning_basic_approach_5e0533.py` | **Best approach**: SFT + Deterministic Solvers + Layer Sensitivity Analysis | 12/52 sensitive layers + shared experts | 9,500 samples |

---

## 2. Hạn chế Kaggle — Không có Internet

Vì `isInternetEnabled: false`, **không thể dùng `pip install` thông thường**. Tất cả package phải được pre-upload lên Kaggle Dataset và cài offline.

### Cách cài offline (dùng trong các file):

```python
# Cách 1 — Unsloth approach (nemotron_unsloth_sft_training.py)
!pip install -q --no-index --find-links \
    /kaggle/input/datasets/mayukh18/nemotron-packages/packages \
    unsloth trl peft transformers datasets accelerate bitsandbytes

# Cách 2 — Offline wheel directory (nvidia_nemotron_tuning_basic_approach_5e0533.py)
!pip install -q --no-index \
    --find-links /kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages \
    datasets trl --ignore-installed

# Cách 3 — Direct .whl file (nvidia_nemotron_sft.py)
!pip install --no-index --no-deps \
    /kaggle/input/datasets/nbroad/hf-libraries/trl/trl-0.29.1-py3-none-any.whl
!pip install -U --no-index --no-deps \
    /kaggle/input/datasets/nbroad/hf-libraries/bitsandbytes/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl
```

> **Lưu ý:** Dataset source tùy notebook — `mayukh18`, `dennisfong`, `nbroad` là các Kaggle user đã upload pre-built packages. Cần kiểm tra dataset nào còn available và tương thích.

### Cách load model (offline):

```python
# Cách 1 — Dùng kagglehub (cần internet hoặc đã cache)
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")

# Cách 2 — Direct path (khuyến nghị khi internet off)
MODEL_PATH = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
```

---

## 3. CUDA Fixes cho Blackwell GPU (RTX PRO 6000)

Blackwell là kiến trúc mới (SM90+), nhiều kernel CUDA của Triton và Nemotron chưa tương thích đầy đủ. Có **4 lỗi chính** cần patch trước khi train.

---

### Fix 1 — RMSNorm C++ Kernel Crash

**Vấn đề:** Kernel Triton-based `rmsnorm_fn` crash trên Blackwell GPU.

**Fix:** Override bằng pure PyTorch implementation:

```python
import torch
import torch.nn.functional as F
import sys

def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                     group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    if upcast:
        x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.float()
    if bias is not None:
        out = out + bias.float()
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype)

# Patch tất cả modules đã được import
for name, mod in list(sys.modules.items()):
    if hasattr(mod, 'rmsnorm_fn'):
        mod.rmsnorm_fn = _pure_rmsnorm_fn
```

> ⚠️ Cần apply patch này **SAU** khi load model, vì `from_pretrained()` có thể import thêm modules mới.

---

### Fix 2 — Triton `ptxas` Permission Error

**Vấn đề:** Triton cần chạy binary `ptxas-blackwell` để compile kernels, nhưng binary trong `/kaggle/usr/lib/...` không có execute permission → silent crash.

**Fix:** Copy toàn bộ `bin/` directory sang `/tmp` và chmod:

```python
import os, stat, shutil
import triton.backends.nvidia as nv_backend
import triton.backends.nvidia.compiler as nv_compiler

# Copy binary ptxas-blackwell riêng lẻ
src = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell"
dst = "/tmp/ptxas-blackwell"
shutil.copy2(src, dst)
os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

# Copy toàn bộ bin directory (để Triton tìm thấy các file phụ thuộc khác)
src_bin = os.path.join(os.path.dirname(nv_backend.__file__), "bin")
dst_bin = "/tmp/triton_nvidia_bin"
shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)

for f in os.listdir(dst_bin):
    fp = os.path.join(dst_bin, f)
    if os.path.isfile(fp):
        os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

# Redirect nv_backend trỏ sang bản copy writable
nv_backend.__file__ = os.path.join(dst_bin, "..", "__init__.py")
os.environ["TRITON_PTXAS_PATH"] = dst
os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
```

---

### Fix 3 — Triton ptxas Version Detection Failure

**Vấn đề:** Triton cố tự detect version của `ptxas` bằng cách gọi subprocess → fail.

**Fix:** Override function bằng hardcoded version:

```python
nv_compiler.get_ptxas_version = lambda arch: "12.0"
```

---

### Fix 4 — Nemotron CUDA Fast Path Broken

**Vấn đề:** Model Nemotron có optimized CUDA fast path cho Mamba layers, nhưng path này crash trên Blackwell. Dễ nhận ra vì có biến `is_fast_path_available` trong source code.

**Fix:** Force slow PyTorch path sau khi load model:

```python
import sys

for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name and hasattr(mod, 'is_fast_path_available'):
        mod.is_fast_path_available = False
        print(f"Patched {name}: is_fast_path_available = False")
```

---

## 4. Workflow chung (Best Approach)

```
Load data (train.csv)
    → Classify puzzle type (6 types: gravity, unit conversion, base conversion, encryption, equation transform, bit manipulation)
    → Run deterministic solvers (gravity, unit conversion, base, encryption)
    → Fallback template CoT cho unsolvable types
    → Format với chat template + system prompt
    → Load model BF16 + Apply Blackwell patches
    → Apply LoRA (rank 32, 12 sensitive layers + shared experts)
    → SFT training (1 epoch, bf16, cosine LR)
    → Save adapter → Package submission.zip
```

---

## 5. LoRA Configuration (Competition Constraints)

- **Max rank:** 32 (hard limit của competition)
- **lora_alpha:** Thường dùng = rank (1x scaling), hoặc 2x (`nvidia_nemotron_sft.py` dùng alpha=64 với r=32)
- **Target modules:** Tùy approach — từ selective (12 sensitive layers) đến `all-linear`
- **gradient_checkpointing:** Bắt buộc để fit VRAM với model 30B

**Submission format:**
```python
import zipfile

with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        zf.write(os.path.join(OUTPUT_DIR, fname), fname)

# Verify
with zipfile.ZipFile("submission.zip") as zf:
    assert "adapter_config.json" in zf.namelist()
```

---

## 6. Các điểm quan trọng cần nhớ

1. **GPU chưa bật**: Tất cả 4 files đều có `isGpuEnabled: false` trong metadata. Cần bật P100/T4 hoặc RTX PRO 6000 khi chạy thực tế.

2. **Apply ALL 4 patches trước khi bất kỳ forward pass nào** — đặc biệt RMSNorm patch cần apply lại sau `from_pretrained()`.

3. **Đường dẫn model** nên dùng direct path `/kaggle/input/models/...` thay vì `kagglehub.model_download()` khi internet tắt, vì `kagglehub` cũng cần network để verify cache.

4. **`offload_folder="/tmp/offload"`** khi load model 30B để tránh OOM (CPU offloading).

5. **Puzzle type "Equation Transformation" và "Bit Manipulation"** chưa có deterministic solver — chỉ dùng fallback template CoT, độ chính xác thấp hơn.

6. **`packing=True`** trong SFTConfig giúp tăng tốc training nhưng cần `max_length` đủ lớn (2048) để không cắt mất `\boxed{}` trong answer.

7. **`gradient_checkpointing_kwargs={"use_reentrant": True}`** — cần thiết cho một số phiên bản transformers cũ trên Kaggle.
