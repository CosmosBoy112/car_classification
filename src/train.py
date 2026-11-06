"""Fine-tune ResNet50 on Stanford Cars.

Example:
    python src/train.py --run_name ft_crop --crop
    python src/train.py --run_name smoke --crop --max_epochs 2      # quick check

Writes to outputs/<run_name>/:
    config.json, train_log.csv (one row per epoch), best.pt (best val top-1),
    predictions_val.csv, probs_val.npy, metrics_val.json
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import config as C
from dataset import CarDataset, make_loader, seed_everything
from evaluate import metrics_from_probs, predict, save_predictions
from models import build_classifier, param_groups


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--crop", action="store_true", help="train and evaluate on bbox crops")
    ap.add_argument("--degrade_aug", action="store_true", help="add random degradation augmentation")
    ap.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--max_epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=C.PATIENCE)
    ap.add_argument("--warmup_epochs", type=float, default=C.WARMUP_EPOCHS)
    ap.add_argument("--lr_backbone", type=float, default=C.LR_BACKBONE)
    ap.add_argument("--lr_head", type=float, default=C.LR_HEAD)
    ap.add_argument("--weight_decay", type=float, default=C.WEIGHT_DECAY)
    ap.add_argument("--num_workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    return ap.parse_args(argv)


def warmup_cosine(step, warmup_steps, total_steps):
    """Linear warmup, then cosine decay to 0. Returns a multiplier on the base LR."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = C.get_device()
    torch.backends.cudnn.benchmark = True  # fixed input size, so autotuning pays off

    run_dir = C.OUTPUTS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args) | {"img_size": C.IMG_SIZE, "bbox_margin": C.BBOX_MARGIN, "rrc_scale": C.RRC_SCALE,
                        "color_jitter": C.COLOR_JITTER, "optimizer": "AdamW", "schedule": "warmup+cosine",
                        "weights": "IMAGENET1K_V2", "device": str(device)}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    train_ds = CarDataset("train", crop=args.crop, train=True, degrade_aug=args.degrade_aug)
    val_ds = CarDataset("val", crop=args.crop, train=False)
    train_dl = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = make_loader(val_ds, 128, shuffle=False, num_workers=args.num_workers)

    model = build_classifier().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(param_groups(model, args.lr_backbone, args.lr_head, args.weight_decay))
    steps_per_epoch = len(train_dl)
    total_steps = args.max_epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, warmup_steps, total_steps))
    loss_fn = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"

    log, best_top1, best_epoch, bad_epochs = [], -1.0, -1, 0
    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        model.train()
        tot_loss, tot_correct, n = 0.0, 0, 0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(y)
            tot_correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        train_time = time.time() - t0

        probs, labels = predict(model, val_dl, device)
        m = metrics_from_probs(probs, labels)
        row = {"epoch": epoch, "train_loss": tot_loss / n, "train_top1": tot_correct / n,
               "val_loss": m["nll"], "val_top1": m["top1"], "val_top5": m["top5"],
               "lr_backbone": opt.param_groups[0]["lr"], "lr_head": opt.param_groups[-1]["lr"],
               "epoch_sec": time.time() - t0, "train_sec": train_time}
        log.append(row)
        pd.DataFrame(log).to_csv(run_dir / "train_log.csv", index=False)
        print(f"[{args.run_name}] epoch {epoch:2d}  train_loss {row['train_loss']:.3f}  "
              f"train_top1 {row['train_top1']:.3f}  val_loss {row['val_loss']:.3f}  "
              f"val_top1 {row['val_top1']:.4f}  {row['epoch_sec']:.0f}s", flush=True)

        if m["top1"] > best_top1:
            best_top1, best_epoch, bad_epochs = m["top1"], epoch, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
            save_predictions(val_ds.df, probs, run_dir / "predictions_val.csv")
            np.save(run_dir / "probs_val.npy", probs.astype(np.float16))
            (run_dir / "metrics_val.json").write_text(json.dumps(m | {"best_epoch": epoch}, indent=2))
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[{args.run_name}] early stop at epoch {epoch}, best epoch {best_epoch}", flush=True)
                break

    print(f"[{args.run_name}] done. best val top-1 {best_top1:.4f} at epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
