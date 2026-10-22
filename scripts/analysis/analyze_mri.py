import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plot_style import (
    AXIS,
    CRITICAL,
    MRI_COLORS,
    SEQ_CMAP,
    SLOTS,
    bar_labels,
    draw_steps,
    headline,
    new_fig,
    save,
)

from src.data.dedup import find_duplicate_pairs, group_ids, thumbnail

DEFAULT_ROOT = Path(
    "data/raw/kagglehub/datasets/indk214/brain-tumor-dataset-segmentation-and-classification/versions/1/DATASET"
)
OUT = Path("reports/figures/data/mri")
METRICS = Path("reports/metrics/data_mri.json")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SEG_DIR_TO_CLASS = {"Glioma": "glioma", "Meningioma": "meningioma", "Pituitary tumor": "pituitary"}
MASK_THRESHOLD = 127
DUP_CORR = 0.98
SEED = 0


@dataclass
class Item:
    source: str
    label: str
    path: Path
    mask: Path | None = None


def index_dataset(root: Path) -> list[Item]:
    items = []
    for split in ["Training", "Testing"]:
        for c in CLASSES:
            items += [Item(split, c, p) for p in sorted((root / "classification" / split / c).glob("*.jpg"))]
    for d, c in SEG_DIR_TO_CLASS.items():
        for p in sorted((root / "Segmentation" / d).glob("enh_*[0-9].png")):
            items.append(Item("Segmentation", c, p, p.with_name(p.stem + "_mask.png")))
    return items


def side_by_side(fig, ax, img: Image.Image, title: str) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10, loc="center")
    ax.set_axis_off()


def overlay(img_path: Path, mask_path: Path, size: int = 256) -> np.ndarray:
    img = np.asarray(Image.open(img_path).convert("RGB").resize((size, size)), dtype=np.float32)
    m = np.asarray(Image.open(mask_path).resize((size, size), Image.NEAREST)) > MASK_THRESHOLD
    out = img.copy()
    out[m] = 0.55 * img[m] + 0.45 * np.array([208, 59, 59])
    return out.astype(np.uint8)


def main(root: Path) -> None:
    random.seed(SEED)
    items = index_dataset(root)
    m: dict = {"root": str(root)}
    print(f"indexed {len(items)} images; reading pixels...")

    sizes, means, t32, t64 = [], [], [], []
    for it in items:
        img = Image.open(it.path).convert("L")
        sizes.append(img.size)
        a = np.asarray(img)
        means.append(float(a.mean()))
        t32.append(thumbnail(img, 32))
        t64.append(thumbnail(img, 64))
    t32, t64 = np.stack(t32), np.stack(t64)
    src = np.array([it.source for it in items])
    lab = np.array([it.label for it in items])

    cls_counts = {s: {c: int(((src == s) & (lab == c)).sum()) for c in CLASSES} for s in ["Training", "Testing"]}
    seg_counts = {c: int(((src == "Segmentation") & (lab == c)).sum()) for c in SEG_DIR_TO_CLASS.values()}
    seg_ids = {
        c: sorted(int(it.path.stem.split("_")[1]) for it in items if it.source == "Segmentation" and it.label == c)
        for c in seg_counts
    }
    m["counts"] = {
        "classification": cls_counts,
        "segmentation_pairs": seg_counts,
        "classification_total": int((src != "Segmentation").sum()),
        "segmentation_file_number_range": {c: [v[0], v[-1]] for c, v in seg_ids.items()},
    }

    fig, (a1, a2) = new_fig(1, 2, gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(4)
    for k, (s, col) in enumerate([("Training", SLOTS[0]), ("Testing", SLOTS[1])]):
        bars = a1.bar(x + (k - 0.5) * 0.38, [cls_counts[s][c] for c in CLASSES], 0.36, color=col, label=s)
        bar_labels(a1, bars)
    a1.set_xticks(x, CLASSES)
    a1.set_title("Classification images (jpg)")
    a1.legend(loc="upper left")
    bars = a2.bar(list(seg_counts), list(seg_counts.values()), 0.6, color=[MRI_COLORS[c] for c in seg_counts])
    bar_labels(a2, bars)
    a2.set_title("Segmentation image + mask pairs (png)")
    headline(
        fig,
        f"{m['counts']['classification_total']:,} classification images + "
        f"{sum(seg_counts.values()):,} segmentation pairs",
        "No 'no tumour' masks exist. Training/Testing folders come from the original Kaggle split",
    )
    save(fig, OUT, "01_counts")

    is512 = np.array([s == (512, 512) for s in sizes])
    fig, ax = new_fig()
    groups_ = [("Segmentation", c) for c in seg_counts] + [("cls", c) for c in CLASSES]
    labels_, vals, cols = [], [], []
    for g, c in groups_:
        sel = (src == "Segmentation") & (lab == c) if g == "Segmentation" else (src != "Segmentation") & (lab == c)
        labels_.append(f"{'seg' if g == 'Segmentation' else 'classif.'}\n{c}")
        vals.append(is512[sel].mean() * 100)
        cols.append(MRI_COLORS[c])
    bars = ax.bar(labels_, vals, color=cols, width=0.6)
    bar_labels(ax, bars, "{:.0f}%")
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of images that are exactly 512 x 512")
    n_sizes_notumor = len({s for s, it in zip(sizes, items) if it.label == "notumor"})
    m["image_sizes"] = {
        "pct_512x512": dict(zip([lbl.replace("\n", " ") for lbl in labels_], np.round(vals, 1).tolist())),
        "distinct_sizes_notumor": n_sizes_notumor,
    }
    headline(
        fig,
        "'No tumour' images look different before the model sees a single tumour",
        f"Tumour images are almost all 512x512; 'no tumour' images come in {n_sizes_notumor} sizes from other "
        "sources\n-> risk: a model can learn 'which source' instead of 'is there a tumour'. Mitigation: resize "
        "everything, test for it",
    )
    save(fig, OUT, "02_image_sizes_shortcut")

    fig, ax = new_fig()
    data = [np.array(means)[(src != "Segmentation") & (lab == c)] for c in CLASSES]
    data.append(np.array(means)[src == "Segmentation"])
    bp = ax.boxplot(
        data,
        tick_labels=[*CLASSES, "segmentation set"],
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "#0b0b0b"},
        flierprops={"markersize": 2, "markeredgecolor": AXIS},
    )
    for patch, col in zip(bp["boxes"], [*[MRI_COLORS[c] for c in CLASSES], "#898781"]):
        patch.set_facecolor(col)
        patch.set_edgecolor(col)
    ax.set_ylabel("mean pixel brightness of the image (0-255)")
    m["mean_brightness_median"] = {
        lbl: round(float(np.median(d)), 1) for lbl, d in zip([*CLASSES, "segmentation"], data)
    }
    med = m["mean_brightness_median"]
    headline(
        fig,
        "Brightness differs by class - another source fingerprint",
        f"Median image brightness: glioma {med['glioma']:.0f}, meningioma {med['meningioma']:.0f}, "
        f"no tumour {med['notumor']:.0f}, pituitary {med['pituitary']:.0f} (0-255). "
        "Per-image normalisation in training removes most of this",
    )
    save(fig, OUT, "03_brightness_by_class")

    fig, axes = plt.subplots(4, 6, figsize=(12, 8.5))
    for r, c in enumerate(CLASSES):
        pool = [it for it in items if it.source != "Segmentation" and it.label == c]
        for k, it in enumerate(random.sample(pool, 6)):
            ax = axes[r, k]
            ax.imshow(Image.open(it.path).convert("L").resize((224, 224)), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if k == 0:
                ax.set_ylabel(c, fontsize=12, rotation=0, ha="right", va="center")
    headline(
        fig,
        "Random classification images, 6 per class",
        "Mixed views (axial, coronal, sagittal) and mixed scanners/sources",
    )
    save(fig, OUT, "04_sample_grid", top=0.9)

    seg_items = [it for it in items if it.source == "Segmentation"]
    patterns, fracs, mismatched = Counter(), {c: [] for c in seg_counts}, 0
    heat = {c: np.zeros((256, 256)) for c in seg_counts}
    for it in seg_items:
        mk = Image.open(it.mask)
        mismatched += mk.size != Image.open(it.path).size
        a = np.asarray(mk)
        u = set(np.unique(a).tolist())
        patterns["{3, 255}" if u == {3, 255} else "{2, 3, 255}" if u == {2, 3, 255} else "other stray values"] += 1
        fracs[it.label].append(float((a > MASK_THRESHOLD).mean()))
        heat[it.label] += np.asarray(mk.resize((256, 256), Image.NEAREST)) > MASK_THRESHOLD
    m["masks"] = {
        "pairs": len(seg_items),
        "size_mismatches": mismatched,
        "value_patterns": dict(patterns),
        "threshold": MASK_THRESHOLD,
        "empty_masks": sum(int(f == 0) for v in fracs.values() for f in v),
        "tumor_pct_of_image": {
            c: {
                "median": round(float(np.median(v)) * 100, 2),
                "p95": round(float(np.percentile(v, 95)) * 100, 2),
                "max": round(float(np.max(v)) * 100, 2),
            }
            for c, v in fracs.items()
        },
    }

    fig, axes = plt.subplots(3, 6, figsize=(12, 7.8))
    for r, c in enumerate(seg_counts):
        for k, it in enumerate(random.sample([s for s in seg_items if s.label == c], 6)):
            axes[r, k].imshow(overlay(it.path, it.mask))
            axes[r, k].set_axis_off()
            axes[r, k].set_title(f"{c} {it.path.stem}", fontsize=8, loc="center")
    headline(
        fig,
        "Visual check: masks sit on the visible lesion",
        f"18 random pairs, red = mask (threshold > {MASK_THRESHOLD}). Inspect by eye for misalignment",
    )
    save(fig, OUT, "05_mask_overlays", top=0.88)

    fig, axes = new_fig(1, 3, sharey=True)
    for ax, c in zip(axes, fracs):
        ax.hist(np.array(fracs[c]) * 100, bins=np.arange(0, 10.5, 0.5), color=MRI_COLORS[c], rwidth=0.85)
        ax.set_title(f"{c} (median {np.median(fracs[c]) * 100:.1f}%)")
        ax.set_xlabel("tumour area, % of image")
    axes[0].set_ylabel("masks")
    headline(
        fig,
        "Tumours are small: typically 1-2% of the image",
        "Area in pixels only - pixel spacing (mm per pixel) is not in the dataset, so no mm2. "
        "Small targets -> use Dice loss, not plain pixel accuracy",
    )
    save(fig, OUT, "06_tumor_area")

    fig, axes = new_fig(1, 3)
    for ax, c in zip(axes, heat):
        ax.imshow(heat[c] / len(fracs[c]), cmap=SEQ_CMAP)
        ax.set_title(c, loc="center")
        ax.set_axis_off()
    headline(
        fig,
        "Where the masks are: a rough anatomy sanity check",
        "Average of all masks per class (darker = more masks cover that pixel). Pituitary masks are compact and "
        "sit near the\nmiddle/lower centre, where the gland is; glioma and meningioma masks spread widely. "
        "Mixed views blur the maps - a sanity check, not proof",
    )
    save(fig, OUT, "07_tumor_location_maps")

    print("all-pairs duplicate search...")
    pairs = find_duplicate_pairs(t32, t64, DUP_CORR)
    kinds = Counter()
    for i, j, _ in pairs:
        si, sj = sorted([items[i].source, items[j].source])
        tag = {
            ("Training", "Training"): "classif. Training <-> Training",
            ("Testing", "Testing"): "classif. Testing <-> Testing",
            ("Testing", "Training"): "classif. Training <-> Testing  (test leak)",
            ("Segmentation", "Segmentation"): "segmentation <-> segmentation",
        }.get((si, sj), "segmentation <-> classif.")
        if items[i].label != items[j].label:
            tag += "  LABEL CONFLICT"
        kinds[tag] += 1
    roots = group_ids(len(items), pairs).tolist()
    sizes_g = Counter(roots)
    in_dup = np.array([sizes_g[r] > 1 for r in roots])
    test_leak = {
        k for k, (i, j, _) in enumerate(pairs) if {items[i].source, items[j].source} == {"Training", "Testing"}
    }
    leaked_test_imgs = {i if items[i].source == "Testing" else j for k, (i, j, _) in enumerate(pairs) if k in test_leak}
    conflict_groups = {roots[i] for i, j, _ in pairs if items[i].label != items[j].label}
    m["duplicates"] = {
        "method": f"Pearson r of 64x64 grey thumbnails > {DUP_CORR}, all pairs",
        "pairs_by_kind": dict(kinds.most_common()),
        "images_in_a_duplicate_group": int(in_dup.sum()),
        "duplicate_groups": int(sum(1 for v in sizes_g.values() if v > 1)),
        "testing_images_with_a_copy_in_training": len(leaked_test_imgs),
        "testing_images_total": int((src == "Testing").sum()),
        "groups_with_label_conflict": len(conflict_groups),
        "images_in_label_conflict_groups": int(sum(sizes_g[g] for g in conflict_groups)),
    }

    fig, ax = new_fig()
    ordered = sorted(kinds.items(), key=lambda kv: kv[1])
    bars = ax.barh(
        [k for k, _ in ordered],
        [v for _, v in ordered],
        color=[CRITICAL if ("CONFLICT" in k or "leak" in k) else SLOTS[0] for k, _ in ordered],
        height=0.6,
    )
    bar_labels(ax, bars, horizontal=True)
    ax.set_xlabel("pairs of near-identical images")
    headline(
        fig,
        f"{len(leaked_test_imgs)} of {int((src == 'Testing').sum()):,} test images also appear in training",
        f"Red = problems. {len(conflict_groups)} groups of identical images carry DIFFERENT tumour labels "
        f"({m['duplicates']['images_in_label_conflict_groups']} images).\nUsing the original Testing folder "
        "as-is would inflate accuracy -> we re-split with duplicate groups kept together",
    )
    save(fig, OUT, "08_duplicates")

    ex = [next((p for p in pairs if {items[p[0]].source, items[p[1]].source} == {"Training", "Testing"}), None)]
    ex += [p for p in pairs if items[p[0]].label != items[p[1]].label][:2]
    ex = [p for p in ex if p]
    fig, axes = plt.subplots(len(ex), 2, figsize=(7, 3.4 * len(ex)))
    for row, (i, j, r) in zip(np.atleast_2d(axes), ex):
        for ax, k, suffix in zip(row, (i, j), ("", f"   (r = {r:.3f})")):
            side_by_side(
                fig,
                ax,
                Image.open(items[k].path).convert("L").resize((256, 256)),
                f"{items[k].source} / {items[k].label}{suffix}\n{items[k].path.name}",
            )
    headline(fig, "Examples: same picture, two places", "Row 1: train/test leak. Rows 2-3: same image, different label")
    save(fig, OUT, "09_duplicate_examples", top=0.9)

    it = seg_items[0]
    raw_m = np.asarray(Image.open(it.mask))
    fig, axes = plt.subplots(1, 5, figsize=(14, 4))
    side_by_side(fig, axes[0], Image.open(it.path).convert("L"), "1. raw image\n512x512, values 0-255")
    axes[1].imshow(raw_m, cmap="gray", vmin=0, vmax=8)
    axes[1].set_title("2. raw mask (contrast boosted)\nbackground = 3, NOT 0", fontsize=10)
    axes[2].imshow(raw_m > 0, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("3. WRONG: mask > 0\n= whole image is 'tumour'", fontsize=10, color=CRITICAL)
    axes[3].imshow(raw_m > MASK_THRESHOLD, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"4. RIGHT: mask > {MASK_THRESHOLD}\n= binary 0/1 mask", fontsize=10)
    axes[4].imshow(overlay(it.path, it.mask))
    axes[4].set_title("5. resize to 256x256\n+ check overlay", fontsize=10)
    for ax in axes:
        ax.set_axis_off()
    headline(
        fig,
        f"Mask preprocessing on a real example ({it.path.parent.name}/{it.path.name})",
        "Getting step 3 wrong would silently train the model on garbage",
    )
    save(fig, OUT, "10_mask_preprocessing_example", top=0.82)

    n_seg_clean = sum(
        1 for k, it in enumerate(items) if it.source == "Segmentation" and roots[k] not in conflict_groups
    )
    notumor_clean = sum(
        1
        for k, it in enumerate(items)
        if it.source != "Segmentation" and it.label == "notumor" and roots[k] not in conflict_groups
    )
    m["preprocessing"] = {
        "seg_pairs_after_dropping_label_conflicts": n_seg_clean,
        "notumor_candidates_after_dropping_conflicts": notumor_clean,
    }
    steps = [
        (
            "Index files",
            f"{len(seg_items):,} segmentation pairs (enh_N.png + enh_N_mask.png), "
            f"{m['counts']['classification_total']:,} classification jpgs",
            "done",
        ),
        ("Verify pairs", f"every mask has its image, same size ({mismatched} mismatches), no empty masks", "done"),
        (
            "Binarise masks",
            f"pixel > {MASK_THRESHOLD} -> 1 (background is stored as 3; stray values in "
            f"{len(seg_items) - patterns['{3, 255}']} masks)",
            "done",
        ),
        (
            "Grayscale + resize",
            "all images -> 1 channel, 256x256 (removes the image-size shortcut); masks resized with nearest-neighbour",
            "done",
        ),
        (
            "Find duplicates",
            f"all-pairs thumbnail correlation > {DUP_CORR}; {len(pairs):,} near-identical pairs "
            f"in {m['duplicates']['duplicate_groups']:,} groups",
            "done",
        ),
        (
            "Drop label-conflict groups",
            f"{m['duplicates']['images_in_label_conflict_groups']} images whose identical "
            f"copy has another label -> {n_seg_clean:,} seg pairs left",
            "done",
        ),
        (
            "Add 'no tumour' images as empty masks",
            f"up to {notumor_clean:,} negatives; report tumour Dice and false alarms separately",
            "done",
        ),
        (
            "Re-split 70/15/15, duplicates together",
            "a duplicate group never spans train and test; no patient IDs, so leaks cannot be fully excluded",
            "done",
        ),
    ]
    draw_steps(
        "MRI data: preprocessing pipeline (segmentation)",
        "Every step is implemented in src/data/brain_mri.py",
        steps,
        OUT,
        "11_preprocessing_steps",
    )

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(m, indent=2))
    print(json.dumps({k: m[k] for k in ["counts", "masks", "duplicates", "preprocessing", "image_sizes"]}, indent=2))
    print("figures ->", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    main(ap.parse_args().root)
