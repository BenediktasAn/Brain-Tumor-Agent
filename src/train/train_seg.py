import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.brain_mri import segmentation_table
from src.data.seg_dataset import SegDataset
from src.models.segmentation import DiceBCELoss, build_unet, image_metrics, remove_small_regions, summarize
from src.utils import resolve_device, run_name, set_seed, write_json


def load_tables(cfg, include_negatives: bool, limit: int | None, seed: int) -> dict[str, pd.DataFrame]:
    path = cfg.paths["processed"] / "mri_index.csv"
    if not path.exists():
        sys.exit(f"{path} missing - run: python scripts/prepare_data.py --only mri")
    table = segmentation_table(pd.read_csv(path, keep_default_na=False), include_negatives)
    out = {s: table[table["split"] == s].reset_index(drop=True) for s in ("train", "val")}
    if limit:
        out = {s: t.sample(min(limit, len(t)), random_state=seed).reset_index(drop=True) for s, t in out.items()}
    return out


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, threshold: float, min_area: int) -> dict:
    model.eval()
    losses, rows = [], []
    for batch in loader:
        x, y = batch["image"].to(device), batch["mask"].to(device)
        with torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
            logits = model(x)
        losses.append(loss_fn(logits.float(), y).item())
        pred = (torch.sigmoid(logits.float()) > threshold).cpu().numpy()[:, 0]
        for p, g in zip(pred, y.cpu().numpy()[:, 0]):
            rows.append(image_metrics(remove_small_regions(p.astype(np.uint8), min_area), g))
    return {"loss": round(float(np.mean(losses)), 4), **summarize(rows)}


def main() -> int:
    cfg = load_config()
    sc = cfg.segmentation
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=sc["epochs"])
    ap.add_argument("--batch-size", type=int, default=sc["batch_size"])
    ap.add_argument("--lr", type=float, default=sc["lr"])
    ap.add_argument("--image-size", type=int, default=cfg.data["image_size"])
    ap.add_argument("--encoder", default=sc["encoder"])
    ap.add_argument("--no-pretrained", action="store_true", help="random encoder init (no ImageNet download)")
    ap.add_argument("--no-negatives", action="store_true", help="train on tumour images only")
    ap.add_argument("--patience", type=int, default=sc["patience"])
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu | mps")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", type=Path, default=None, help="path to a last.pt to continue from")
    ap.add_argument("--limit", type=int, default=None, help="use at most N images per split (smoke runs)")
    ap.add_argument("--dry-run", action="store_true", help="build everything, one forward pass, no training")
    args = ap.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    tables = load_tables(cfg, not args.no_negatives, args.limit, args.seed)
    for s, t in tables.items():
        print(f"{s}: {len(t)} images ({int((t['mask_path'] != '').sum())} with tumour) on {device}")
    thr, min_area = sc["threshold"], sc["min_area_px"]
    ds = {
        s: SegDataset(t, args.image_size, cfg.data["mask_threshold"], train=(s == "train")) for s, t in tables.items()
    }
    loaders = {
        s: DataLoader(
            d,
            batch_size=args.batch_size,
            shuffle=(s == "train"),
            num_workers=args.num_workers,
            pin_memory=device.startswith("cuda"),
            drop_last=(s == "train" and len(d) > args.batch_size),
        )
        for s, d in ds.items()
    }
    model = build_unet(args.encoder, pretrained=not args.no_pretrained).to(device)
    loss_fn = DiceBCELoss()

    if args.dry_run:
        batch = next(iter(loaders["val"]))
        model.eval()
        with torch.no_grad():
            logits = model(batch["image"].to(device))
        print(
            f"dry run OK: input {tuple(batch['image'].shape)} -> logits {tuple(logits.shape)}, "
            f"loss {loss_fn(logits, batch['mask'].to(device)).item():.3f}; nothing was trained"
        )
        return 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))
    start, best, history, stale = 0, -1.0, [], 0
    run = args.run_name or run_name("seg")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start, best, history, stale, run = ck["epoch"] + 1, ck["best"], ck["history"], ck["stale"], ck["run"]
        print(f"resumed {run} at epoch {start}")
    out = cfg.paths["models"] / "segmentation" / run
    model_cfg = {"encoder": args.encoder, "image_size": args.image_size, "threshold": thr, "min_area_px": min_area}
    write_json(
        out / "config.json",
        {
            **vars(args),
            "device": device,
            "run": run,
            "model": model_cfg,
            "rows": {s: len(t) for s, t in tables.items()},
        },
    )

    for epoch in range(start, args.epochs):
        model.train()
        t0, losses = time.time(), []
        for batch in loaders["train"]:
            x, y = batch["image"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
                loss = loss_fn(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        val = evaluate(model, loaders["val"], loss_fn, device, thr, min_area)
        rec = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 4),
            "val": val,
            "lr": opt.param_groups[0]["lr"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        score = val.get("val_score", val.get("dice_mean", 0.0))
        improved = score > best
        if improved:
            best, stale = score, 0
            torch.save(
                {"state_dict": model.state_dict(), "config": model_cfg, "epoch": epoch, "val": val}, out / "best.pt"
            )
        else:
            stale += 1
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best": best,
                "history": history,
                "stale": stale,
                "run": run,
                "config": model_cfg,
            },
            out / "last.pt",
        )
        write_json(out / "history.json", history)
        print(
            f"epoch {epoch + 1}/{args.epochs} train_loss={rec['train_loss']:.4f} val_loss={val['loss']:.4f} "
            f"dice={val.get('dice_mean', float('nan')):.4f} "
            f"false_alarm={val.get('false_alarm_rate', float('nan')):.3f} "
            f"{'*best*' if improved else ''} ({rec['seconds']}s)"
        )
        if stale >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs")
            break

    if not (out / "best.pt").exists():
        print("no checkpoint was saved (zero epochs?)")
        return 1
    cfg.paths["segmentation_model"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / "best.pt", cfg.paths["segmentation_model"])
    best_rec = max(history, key=lambda r: r["val"].get("val_score", r["val"].get("dice_mean", 0)))
    write_json(cfg.paths["metrics"] / f"seg_train_{run}.json", {"run": run, "best_epoch": best_rec, "history": history})
    print(f"best val_score={best:.4f}; saved {out} and copied best.pt to {cfg.paths['segmentation_model']}")
    print("next: python -m src.eval.evaluate_seg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
