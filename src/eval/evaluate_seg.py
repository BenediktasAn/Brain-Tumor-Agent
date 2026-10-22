import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.brain_mri import load_image, segmentation_table
from src.data.seg_dataset import SegDataset
from src.models.segmentation import Segmenter, image_metrics, remove_small_regions, summarize
from src.utils import resolve_device, write_json


def overlay_gt_pred(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    rgb = np.repeat((np.clip(gray, 0, 1) * 255)[..., None], 3, axis=2).astype(np.float32)
    rgb[pred.astype(bool)] = 0.5 * rgb[pred.astype(bool)] + 0.5 * np.array([208, 59, 59])
    g = gt.astype(bool)
    rgb[g & ~ndimage.binary_erosion(g, iterations=2)] = [12, 163, 12]
    return rgb.astype(np.uint8)


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=cfg.paths["segmentation_model"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = resolve_device(args.device)
    seg = Segmenter.load(args.checkpoint, device)
    mc = seg.cfg
    index = pd.read_csv(cfg.paths["processed"] / "mri_index.csv", keep_default_na=False)
    table = segmentation_table(index, include_negatives=True)
    table = table[table["split"] == args.split].reset_index(drop=True)
    ds = SegDataset(table, mc["image_size"], cfg.data["mask_threshold"], train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)

    rows, keep = [], {}
    with torch.no_grad():
        for batch in loader:
            prob = torch.sigmoid(seg.model(batch["image"].to(device))).cpu().numpy()[:, 0]
            for k in range(len(prob)):
                pred = remove_small_regions((prob[k] > mc["threshold"]).astype(np.uint8), mc["min_area_px"])
                gt = batch["mask"][k, 0].numpy()
                r = image_metrics(pred, gt) | {"path": batch["path"][k], "label": batch["label"][k]}
                rows.append(r)
                keep[r["path"]] = (pred, gt)

    per_image = pd.DataFrame(rows)
    summary = {"checkpoint": str(args.checkpoint), "split": args.split, "model": mc, **summarize(rows)}
    summary["per_tumour_type"] = {
        lab: summarize(g.to_dict("records")) for lab, g in per_image[per_image["gt_area"] > 0].groupby("label")
    }
    out = write_json(cfg.paths["metrics"] / f"seg_eval_{args.split}.json", summary)
    per_image.to_csv(cfg.paths["metrics"] / f"seg_eval_{args.split}_per_image.csv", index=False)

    figs = cfg.paths["figures"] / "models"
    figs.mkdir(parents=True, exist_ok=True)
    pos = per_image[per_image["gt_area"] > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(pos["dice"], bins=np.linspace(0, 1, 21), color="#2a78d6", rwidth=0.85)
    ax.set_xlabel("Dice per tumour image")
    ax.set_ylabel("images")
    ax.set_title(f"Segmentation Dice on {args.split} ({len(pos)} tumour images, mean {pos['dice'].mean():.3f})")
    fig.tight_layout()
    fig.savefig(figs / f"seg_dice_hist_{args.split}.png", dpi=150)
    plt.close(fig)

    rng = np.random.default_rng(cfg.seed)
    picks = []
    for lab, g in per_image.groupby("label"):
        picks += [(f"{lab}", p) for p in rng.choice(g["path"], size=min(3, len(g)), replace=False)]
    picks += [("worst", p) for p in pos.nsmallest(3, "dice")["path"]]
    fig, axes = plt.subplots(3, (len(picks) + 2) // 3, figsize=(16, 9))
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, (tag, path) in zip(axes.ravel(), picks):
        img = load_image(Path(path), size=mc["image_size"])
        pred, gt = keep[path]
        r = per_image.loc[per_image["path"] == path].iloc[0]
        ax.imshow(overlay_gt_pred(img, gt, pred))
        ax.set_title(
            f"{tag}: Dice {r['dice']:.2f}"
            if r["gt_area"]
            else f"{tag}: {'FALSE ALARM' if r['pred_area'] else 'clear'}",
            fontsize=9,
        )
    fig.suptitle(f"Segmentation on {args.split}: green outline = ground truth, red = prediction", fontsize=12)
    fig.tight_layout()
    fig.savefig(figs / f"seg_examples_{args.split}.png", dpi=130)
    plt.close(fig)

    print({k: v for k, v in summary.items() if k not in ("per_tumour_type", "model")})
    for lab, s in summary["per_tumour_type"].items():
        print(f"  {lab}: dice_mean={s.get('dice_mean')} n={s['n_tumour_images']}")
    print(f"-> {out}, figures in {figs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
