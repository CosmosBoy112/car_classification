"""Copy the chosen sweep runs to outputs/ft_full and outputs/ft_crop.

The sweep run at the chosen learning rate has exactly the same config and seed as a separate
fine-tuning run would, so it is reused instead of training again (docs/decisions.md).
The copied config.json gets the new run_name and a "copied_from" field.

    python src/promote_sweep.py              # uses config.LR_BACKBONE (1e-4)
    python src/promote_sweep.py --lr 3e-4
"""
import argparse
import json
import shutil

import config as C

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=C.LR_BACKBONE)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for inp in ("full", "crop"):
        src = C.OUTPUTS_DIR / f"sweep_{inp}_lr{args.lr:g}"
        dst = C.OUTPUTS_DIR / f"ft_{inp}"
        if not src.exists():
            raise SystemExit(f"{src} not found, run src/sweep_lr.py first")
        if dst.exists():
            if not args.overwrite:
                raise SystemExit(f"{dst} already exists, pass --overwrite to replace it")
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        cfg = json.loads((dst / "config.json").read_text())
        cfg["run_name"], cfg["copied_from"] = dst.name, src.name
        (dst / "config.json").write_text(json.dumps(cfg, indent=2))
        print(f"{src.name} -> {dst.name}")
