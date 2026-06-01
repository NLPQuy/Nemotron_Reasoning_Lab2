# >>> EXP17 START
import os
import shutil
import sys

import safetensors.torch as st


def main() -> None:
    paths = sys.argv[1:-1]
    out = sys.argv[-1] if len(sys.argv) >= 3 else None
    if not paths or out is None:
        raise SystemExit("usage: python soup_adapters.py ADAPTER_DIR... OUT_DIR")

    tensors = [st.load_file(f"{p}/adapter_model.safetensors") for p in paths]
    keys = set(tensors[0])
    assert all(set(t) == keys for t in tensors), "key mismatch"
    avg = {k: sum(t[k].float() for t in tensors) / len(tensors) for k in keys}
    os.makedirs(out, exist_ok=True)
    st.save_file(avg, f"{out}/adapter_model.safetensors")
    shutil.copy(f"{paths[0]}/adapter_config.json", f"{out}/adapter_config.json")
    print(f"souped {len(paths)} adapters -> {out}")


if __name__ == "__main__":
    main()
# <<< EXP17 END
