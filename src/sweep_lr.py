"""Small learning-rate sweep on val, for both input types.

The head LR is always 10x the backbone LR. Results go to outputs/sweep_<input>_lr<lr>/
and the summary to outputs/sweep_lr_summary.csv.

    python src/sweep_lr.py
"""
import json

import pandas as pd

import config as C
import train

BACKBONE_LRS = [3e-5, 1e-4, 3e-4]

if __name__ == "__main__":
    rows = []
    for crop in (False, True):
        for lr in BACKBONE_LRS:
            name = f"sweep_{'crop' if crop else 'full'}_lr{lr:g}"
            argv = ["--run_name", name, "--lr_backbone", str(lr), "--lr_head", str(10 * lr)]
            train.main(argv + (["--crop"] if crop else []))
            m = json.loads((C.OUTPUTS_DIR / name / "metrics_val.json").read_text())
            rows.append({"input": "crop" if crop else "full", "lr_backbone": lr, "lr_head": 10 * lr,
                         "val_top1": m["top1"], "val_top5": m["top5"], "val_nll": m["nll"],
                         "best_epoch": m["best_epoch"]})
            pd.DataFrame(rows).to_csv(C.OUTPUTS_DIR / "sweep_lr_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
