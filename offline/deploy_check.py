#!/usr/bin/env python3
"""Deploy-check (plan-batch-6 PHẦN 2C) for a LoRA adapter dir.

Verifies a trained adapter is a clean rank-32 vLLM-loadable LoRA:
  - peft_type == LORA, r <= 32, no modules_to_save
  - every tensor key is lora_A / lora_B / lora_embedding_A / lora_embedding_B
    (the lm_head base_layer.weight present in the 0.86 baseline is tolerated;
     any OTHER full base weight — e.g. embed_tokens.base_layer.weight — is FLAGGED)

Usage:  python3 offline/deploy_check.py <adapter_dir>
No heavy deps: reads the safetensors header directly.
"""
import collections
import json
import re
import struct
import sys

# Full base weights that the known-good 0.86 baseline already ships (tolerated).
ALLOWED_BASE_WEIGHTS = {"base_model.model.backbone.lm_head.base_layer.weight"}
LORA_SUFFIXES = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")


def read_safetensors_keys(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return {k: v for k, v in hdr.items() if k != "__metadata__"}


def main(adapter_dir: str) -> int:
    cfg = json.load(open(f"{adapter_dir}/adapter_config.json"))
    hdr = read_safetensors_keys(f"{adapter_dir}/adapter_model.safetensors")

    problems = []
    print(f"== deploy-check: {adapter_dir} ==")
    print(f"peft_type={cfg.get('peft_type')} r={cfg.get('r')} "
          f"use_rslora={cfg.get('use_rslora')}")
    print(f"target_modules={cfg.get('target_modules')}")
    print(f"modules_to_save={cfg.get('modules_to_save')}")

    if cfg.get("peft_type") != "LORA":
        problems.append(f"peft_type != LORA ({cfg.get('peft_type')})")
    if (cfg.get("r") or 0) > 32:
        problems.append(f"r > 32 ({cfg.get('r')})")
    if cfg.get("modules_to_save"):
        problems.append(f"modules_to_save set: {cfg.get('modules_to_save')} "
                        "(vLLM ValueError)")

    cats = collections.Counter()
    ranks = set()
    for k, meta in hdr.items():
        m = re.search(r"(lora_embedding_[AB]|lora_[AB])", k)
        if m:
            cats[m.group(1)] += 1
            shp = meta["shape"]
            if m.group(1) in ("lora_A", "lora_embedding_A"):
                ranks.add(min(shp))
            else:
                ranks.add(min(shp))
        elif k in ALLOWED_BASE_WEIGHTS:
            cats["ALLOWED_base_layer"] += 1
        else:
            cats["UNEXPECTED"] += 1
            problems.append(f"unexpected key (not lora_*, not allowed base): {k}")

    print(f"key categories: {dict(cats)}")
    print(f"observed ranks: {sorted(ranks)}")
    if ranks and max(ranks) > 32:
        problems.append(f"a lora tensor has rank > 32: {sorted(ranks)}")

    if problems:
        print("\nFAIL — not deploy-clean:")
        for p in problems:
            print(f"  ✗ {p}")
        return 1
    print("\nPASS — clean rank-32 vLLM LoRA adapter.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "submission"))
