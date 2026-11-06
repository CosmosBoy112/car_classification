"""Predictions and metrics.

Library use:   predict(model, loader, device) -> probs, labels
               metrics_from_probs(probs, labels) -> dict
               save_predictions(df, probs, path)

CLI (val):     python src/evaluate.py --run_name ft_crop --split val
CLI (test):    python src/evaluate.py --run_name final --split test --allow_test
The test split needs an explicit flag so it cannot be evaluated by accident.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

import config as C
from dataset import CarDataset, make_loader


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), 1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def metrics_from_probs(probs: np.ndarray, labels: np.ndarray) -> dict:
    top5 = np.argsort(-probs, 1)[:, :5]
    pred = top5[:, 0]
    correct = pred == labels
    per_class = pd.Series(correct).groupby(labels).mean()
    nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean()
    return {
        "n": int(len(labels)),
        "top1": float(correct.mean()),
        "top5": float((top5 == labels[:, None]).any(1).mean()),
        "mean_per_class_acc": float(per_class.mean()),
        "worst_class_acc": float(per_class.min()),
        "nll": float(nll),
        "mean_confidence": float(probs.max(1).mean()),
    }


def save_predictions(df: pd.DataFrame, probs: np.ndarray, path):
    top5 = np.argsort(-probs, 1)[:, :5]
    out = pd.DataFrame({
        "path": df.path.values,
        "label": df.label.values,
        "pred": top5[:, 0],
        "confidence": probs.max(1),
        "correct": top5[:, 0] == df.label.values,
    })
    for k in range(5):
        out[f"top{k + 1}"] = top5[:, k]
        out[f"top{k + 1}_prob"] = probs[np.arange(len(probs)), top5[:, k]]
    out.to_csv(path, index=False)
    return out


def evaluate_run(run_name, split="val", degrade=None, allow_test=False, tag=None, batch_size=128):
    if split == "test" and not allow_test:
        raise SystemExit("Refusing to evaluate on test without --allow_test (test is used once, at the end).")
    run_dir = C.OUTPUTS_DIR / run_name
    cfg = json.loads((run_dir / "config.json").read_text())
    device = C.get_device()
    from models import load_checkpoint
    model = load_checkpoint(run_dir / "best.pt", device).to(memory_format=torch.channels_last)
    ds = CarDataset(split, crop=cfg["crop"], train=False, degrade=degrade)
    probs, labels = predict(model, make_loader(ds, batch_size, shuffle=False), device)
    m = metrics_from_probs(probs, labels)
    tag = tag or split
    save_predictions(ds.df, probs, run_dir / f"predictions_{tag}.csv")
    np.save(run_dir / f"probs_{tag}.npy", probs.astype(np.float16))
    (run_dir / f"metrics_{tag}.json").write_text(json.dumps(m, indent=2))
    return m, probs, labels


# Conditions for the robustness experiment. Each family has its clean point first.
DEGRADATIONS = {
    "resolution": [("res224", None), ("res112", {"res": 112}), ("res64", {"res": 64}), ("res32", {"res": 32})],
    "blur": [("blur0", None), ("blur1", {"blur": 1.0}), ("blur2", {"blur": 2.0}), ("blur3", {"blur": 3.0}),
             ("blur4", {"blur": 4.0})],
    "dark": [("dark1.0", None), ("dark0.5", {"dark": 0.5}), ("dark0.3", {"dark": 0.3}),
             ("dark0.15", {"dark": 0.15})],
}


def evaluate_degradations(run_name, split="val", allow_test=False, threshold=None):
    """Accuracy under each degradation. Also coverage and accepted accuracy if a threshold is given."""
    rows, cache = [], {}
    for family, conds in DEGRADATIONS.items():
        for cond, degrade in conds:
            key = json.dumps(degrade)
            if key not in cache:
                tag = f"{split}_{cond}" if degrade else split
                cache[key] = evaluate_run(run_name, split, degrade, allow_test=allow_test, tag=tag)
            m, probs, labels = cache[key]
            level = 224 if cond == "res224" else float(cond.lstrip("resbludark") or 0)
            row = {"run": run_name, "split": split, "family": family, "condition": cond, "level": level,
                   "top1": m["top1"], "top5": m["top5"], "mean_confidence": m["mean_confidence"]}
            if threshold is not None:
                conf = probs.max(1)
                acc = conf >= threshold
                row["coverage"] = float(acc.mean())
                row["accepted_top1"] = float((probs.argmax(1) == labels)[acc].mean()) if acc.any() else float("nan")
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(C.OUTPUTS_DIR / run_name / f"degradation_{split}.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--allow_test", action="store_true")
    args = ap.parse_args()
    m, _, _ = evaluate_run(args.run_name, args.split, allow_test=args.allow_test)
    print(json.dumps(m, indent=2))
