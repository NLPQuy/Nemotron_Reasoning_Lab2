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


# >>> EXP_WISEFT START
# WiSE-FT weighted interpolation (D7 — Batch-5)
# usage: python soup_adapters.py --alpha 0.2 θ0_dir θ1_dir out_dir
# θ = (1-alpha)*θ0 + alpha*θ1 ; θ0 = 0.86 anchor, θ1 = lever (vd exp35/exp38)
def wiseft(alpha: float, path0: str, path1: str, out: str) -> None:
    t0 = st.load_file(f"{path0}/adapter_model.safetensors")
    t1 = st.load_file(f"{path1}/adapter_model.safetensors")
    keys = set(t0)
    assert set(t1) == keys, "key mismatch between θ0 and θ1"
    avg = {k: (1 - alpha) * t0[k].float() + alpha * t1[k].float() for k in keys}
    os.makedirs(out, exist_ok=True)
    st.save_file(avg, f"{out}/adapter_model.safetensors")
    shutil.copy(f"{path0}/adapter_config.json", f"{out}/adapter_config.json")
    print(f"WiSE-FT α={alpha}: (1-α)*{path0} + α*{path1} -> {out}")
# >>> EXP_WISEFT END


if __name__ == "__main__":
    # >>> EXP_WISEFT START
    if "--alpha" in sys.argv:
        idx = sys.argv.index("--alpha")
        alpha = float(sys.argv[idx + 1])
        rest = sys.argv[1:idx] + sys.argv[idx + 2:]
        if len(rest) != 3:
            raise SystemExit(
                "usage: python soup_adapters.py --alpha <α> θ0_dir θ1_dir out_dir"
            )
        wiseft(alpha, rest[0], rest[1], rest[2])
    else:
        main()
    # >>> EXP_WISEFT END
# <<< EXP17 END
