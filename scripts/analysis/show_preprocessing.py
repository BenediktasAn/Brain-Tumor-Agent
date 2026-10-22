from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from plot_style import MRI_COLORS, SLOTS, draw_steps, headline, new_fig, save, table_fig

from src.config import load_config
from src.data import headache as H
from src.data.brain_mri import load_image, segmentation_table
from src.data.seg_dataset import SegDataset
from src.models.headache_clf import build_pipeline, mask_unknowns
from src.models.segmentation import overlay

CFG = load_config()
OUT = CFG.paths["figures"] / "preprocessing"
SIZE = CFG.data["image_size"]


def seg_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.read_csv(CFG.paths["processed"] / "mri_index.csv", keep_default_na=False)
    return index, segmentation_table(index, CFG.data["seg_include_negatives"])


def fig_model_input(table: pd.DataFrame) -> None:
    train = table[table["split"] == "train"]
    picks = [train[train["mask_path"] != ""].iloc[0], train[train["mask_path"] == ""].iloc[1]]
    ds = SegDataset(pd.DataFrame(picks), SIZE, CFG.data["mask_threshold"], train=False)
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.2), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
    for r in range(2):
        raw_img, raw_mask = ds.raw(r)
        sample = ds[r]
        x, m = sample["image"][0].numpy(), sample["mask"][0].numpy()
        resized = torch.nn.functional.interpolate(
            torch.from_numpy(raw_img)[None, None], size=(SIZE, SIZE), mode="bilinear", antialias=True
        )[0, 0].numpy()
        kind = "tumour image" if picks[r]["mask_path"] else "healthy scan (empty mask)"
        panels = [
            (raw_img, f"1. raw {kind}\n{raw_img.shape[1]}x{raw_img.shape[0]}, values 0-1", "gray", (0, 1)),
            (resized, f"2. resized to {SIZE}x{SIZE}", "gray", (0, 1)),
            (m, f"3. mask, binarised (> {CFG.data['mask_threshold']})\n{int(m.sum())} tumour pixels", "gray", (0, 1)),
            (x, "4. standardised per image\n(mean 0, std 1) = model input", "gray", (-2.5, 2.5)),
        ]
        for ax, (img, title, cmap, (lo, hi)) in zip(axes[r], panels):
            ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(title, fontsize=10)
            ax.set_axis_off()
        ax = axes[r, 4]
        ax.hist(resized.ravel(), bins=60, color=SLOTS[1], alpha=0.8, label="before (0-1)")
        ax.hist(x.ravel(), bins=60, color=SLOTS[0], alpha=0.8, label="after (z-score)")
        ax.set_yscale("log")
        ax.set_title("5. pixel values before / after", fontsize=10)
        ax.legend(fontsize=8)
    headline(
        fig,
        "What the segmentation model sees",
        f"Top: tumour image ({picks[0]['label']}). Bottom: healthy scan "
        "used as an empty-mask example. Per-image standardisation removes brightness differences between sources",
    )
    save(fig, OUT, "01_seg_model_input", top=0.86)


def fig_augmentations(table: pd.DataFrame) -> None:
    row = table[(table["split"] == "train") & (table["mask_path"] != "")].iloc[[5]]
    plain = SegDataset(row, SIZE, CFG.data["mask_threshold"], train=False)[0]
    aug = SegDataset(row, SIZE, CFG.data["mask_threshold"], train=True)
    torch.manual_seed(CFG.seed)
    samples = [plain] + [aug[0] for _ in range(7)]

    def show(s: dict) -> np.ndarray:
        x = s["image"][0].numpy()
        return overlay((x - x.min()) / (x.max() - x.min() + 1e-6), s["mask"][0].numpy())

    fig, axes = plt.subplots(2, 4, figsize=(14, 8.6))
    for k, (ax, s) in enumerate(zip(axes.ravel(), samples)):
        ax.imshow(show(s))
        ax.set_title(
            "original (validation/test look like this)" if k == 0 else f"random training version {k}", fontsize=10
        )
        ax.set_axis_off()
    headline(
        fig,
        "Training augmentation: the mask moves with the image",
        "Random flip, rotation (±10°), shift (±5%), zoom (90-110%) and brightness/contrast (±20%). "
        "Only during training; red = tumour mask",
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.85, hspace=0.12, wspace=0.04)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "02_seg_augmentation.png", dpi=160)
    plt.close(fig)


def fig_healthy_scans(table: pd.DataFrame) -> None:
    usable = table[table["split"] != "excluded"]
    rows = [
        ("healthy ('notumor')", usable[usable["mask_path"] == ""].sample(8, random_state=1)),
        ("tumour (segmentation set)", usable[usable["mask_path"] != ""].sample(8, random_state=1)),
    ]
    fig, axes = plt.subplots(2, 8, figsize=(16, 5.6))
    for r, (name, sample) in enumerate(rows):
        for ax, (_, row) in zip(axes[r], sample.iterrows()):
            ax.imshow(np.asarray(load_image(Path(row["path"]), size=SIZE)), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(name, fontsize=11)
    headline(
        fig,
        "Caveat: healthy and tumour images are different kinds of scan",
        "Tumour images are T1 with contrast (dark fluid). Most healthy images are T2/FLAIR (bright fluid) and "
        "about 1 in 12 look like CT.\nThe model may partly learn 'scan type' instead of 'tumour', so the "
        "false-alarm rate on healthy test scans is likely optimistic",
    )
    save(fig, OUT, "08_healthy_vs_tumour_scan_types", top=0.83)


def fig_seg_steps(table: pd.DataFrame) -> None:
    tr = table[table["split"] == "train"]
    steps = [
        (
            "Select rows",
            f"{int((tr['mask_path'] != '').sum()):,} tumour pairs + {int((tr['mask_path'] == '').sum()):,} "
            "healthy scans in train (label-conflict images excluded)",
            "done",
        ),
        ("Load", "image -> grayscale, values 0-1;  mask -> binary at > 127 (healthy scan -> all-zero mask)", "done"),
        ("Resize", f"{SIZE}x{SIZE}: bilinear for the image, nearest-neighbour for the mask (stays 0/1)", "done"),
        (
            "Augment (train only)",
            "flip, rotate ±10°, shift ±5%, zoom 90-110% on image+mask together; "
            "brightness/contrast ±20% on the image only",
            "done",
        ),
        ("Standardise", "per image: subtract mean, divide by std (removes source brightness differences)", "done"),
        (
            "Batch",
            f"{CFG.segmentation['batch_size']} images x 1 channel x {SIZE} x {SIZE} -> U-Net "
            f"({CFG.segmentation['encoder']} encoder, ImageNet weights)",
            "done",
        ),
        (
            "Post-process predictions",
            f"probability > {CFG.segmentation['threshold']}; drop regions < "
            f"{CFG.segmentation['min_area_px']} px; area in pixels and % of image (no mm²)",
            "done",
        ),
    ]
    draw_steps(
        "Segmentation: from file to model input",
        "Implemented in src/data/seg_dataset.py and src/models/segmentation.py",
        steps,
        OUT,
        "03_seg_pipeline_steps",
    )


def fig_splits(table: pd.DataFrame) -> None:
    hd = pd.read_csv(CFG.paths["processed"] / "headache_clean.csv", keep_default_na=False)
    fig, (a1, a2) = new_fig(1, 2)
    splits = ["train", "val", "test"]
    bottom = np.zeros(3)
    for lab in ["glioma", "meningioma", "pituitary", "notumor"]:
        sel = table["label"] == lab
        v = np.array([int((sel & (table["split"] == s)).sum()) for s in splits])
        a1.bar(
            splits,
            v,
            bottom=bottom,
            color=MRI_COLORS[lab],
            width=0.6,
            label="healthy (empty mask)" if lab == "notumor" else lab,
            edgecolor="white",
            linewidth=1.5,
        )
        bottom += v
    for i, b in enumerate(bottom):
        a1.annotate(f"{int(b):,}", (i, b), xytext=(0, 3), textcoords="offset points", ha="center")
    a1.set_title("Segmentation images")
    a1.legend(fontsize=9)
    hs = ["train", "test"]
    real = [int(((hd["split"] == s) & (hd["is_original"].astype(str) == "True")).sum()) for s in hs]
    total = [int((hd["split"] == s).sum()) for s in hs]
    a2.bar(hs, real, color=SLOTS[0], width=0.5, label="real patients", edgecolor="white", linewidth=1.5)
    a2.bar(
        hs,
        np.array(total) - real,
        bottom=real,
        color="#898781",
        width=0.5,
        label="generated rows",
        edgecolor="white",
        linewidth=1.5,
    )
    for i, t in enumerate(total):
        a2.annotate(f"{t:,} ({real[i]} real)", (i, t), xytext=(0, 3), textcoords="offset points", ha="center")
    a2.set_title("Headache rows")
    a2.legend(fontsize=9)
    headline(
        fig,
        "Final datasets after preprocessing",
        "MRI: 70/15/15 split, near-duplicate copies always in the "
        "same split. Headache: 80/20 stratified split of the 2,748 unique rows",
    )
    save(fig, OUT, "04_final_splits")


def fig_headache_input(hd: pd.DataFrame) -> None:
    said = {
        "location": "unilateral",
        "characterisation": "throbbing",
        "durationGroup": H.duration_group(6 * 3600),
        "headache_days": 3,
        "nausea": "yes",
        "photophobia": "yes",
        "phonophobia": "no",
        "motor_weakness": "no",
        "speech_disturbance": "no",
    }
    quotes = {
        "location": '"only on the left side"',
        "characterisation": '"throbbing"',
        "durationGroup": '"about 6 hours" (code H = 4 h to 3 days)',
        "headache_days": '"3 days this month"',
        "nausea": '"I feel sick"',
        "photophobia": '"light hurts"',
        "phonophobia": '"noise is fine"',
        "motor_weakness": '"no weakness"',
        "speech_disturbance": '"no"',
    }
    x = H.record_to_features(said).iloc[0]
    pre = build_pipeline("random_forest", H.FEATURES, CFG.seed).named_steps["pre"]
    pre.fit(hd[hd["split"] == "train"][H.FEATURES])
    vec = pre.transform(H.record_to_features(said))[0]
    names = pre.get_feature_names_out()
    rows, hl = [], {}
    show = [
        "location",
        "characterisation",
        "durationGroup",
        "headache_days",
        "nausea",
        "photophobia",
        "phonophobia",
        "vomitting",
        "severity",
        "aura_duration",
    ]
    for c in show:
        on = [n for n, v in zip(names, vec) if n.startswith(c + "_") and v == 1]
        enc = ", ".join(on) if c not in H.NUMERIC else (f"headache_days = {x[c]:.0f}, missing-flag = 0")
        rows.append([c, quotes.get(c, "(not mentioned)"), str(x[c]), enc])
        if c not in quotes:
            hl[len(rows) - 1] = "#fdf3dc"
    rows.append(["... 20 more symptom fields", "(not mentioned)", "unknown", "<field>_unknown = 1"])
    hl[len(rows) - 1] = "#fdf3dc"
    rows.append(["red flags (e.g. motor_weakness)", '"no weakness"', "no", "NOT a model input -> red-flag rule"])
    hl[len(rows) - 1] = "#f6d6d6"
    table_fig(
        rows,
        ["field", "what the patient said (illustrative)", "cleaned value", "model input (one-hot)"],
        "Headache classifier: from conversation to model input",
        f"{len(H.FEATURES)} fields -> {len(vec)} numbers. "
        "Amber = not mentioned, becomes 'unknown' (never a guessed 'no').\n"
        "Red = red flags go to the scan rule, never to the classifier",
        "05_headache_model_input",
        hl,
        col_widths=[0.24, 0.32, 0.12, 0.32],
        out_dir=OUT,
    )


def fig_unknown_masking(hd: pd.DataFrame) -> None:
    row = hd[(hd["split"] == "train")].iloc[[3]]
    masked = mask_unknowns(row, 0.5, np.random.default_rng(CFG.seed), H.FEATURES)
    show = H.FEATURES[:14]
    rows, hl = [], {}
    for c in show:
        a, b = row.iloc[0][c], masked.iloc[0][c]
        a, b = (f"{v:g}" if isinstance(v, int | float) and not pd.isna(v) else v for v in (a, b))
        b = "unknown (empty)" if pd.isna(b) else b
        rows.append([c, str(a), str(b)])
        if str(a) != str(b):
            hl[len(rows) - 1] = "#fdf3dc"
    table_fig(
        rows,
        ["field", "training row", "extra masked copy"],
        "Training trick: extra copies with fields blanked out",
        f"Each training row gets one copy where every field is blanked with probability "
        f"{CFG.headache_model['unknown_augment']:.0%} (shown here at 50% for visibility).\nThe chatbot rarely "
        "learns every field, so the model must cope with 'unknown'. Validation/test rows are never masked",
        "06_headache_unknown_masking",
        hl,
        col_widths=[0.34, 0.33, 0.33],
        out_dir=OUT,
    )


def fig_headache_steps() -> None:
    steps = [
        ("Cleaned table", "2,748 unique rows from src/data/headache.py (see the data-preprocessing slide)", "done"),
        ("Choose inputs", f"{len(H.FEATURES)} fields; the 8 red flags and previous_attacks are excluded", "done"),
        ("Encode", "every categorical / yes-no field -> one-hot over a FIXED list that includes 'unknown'", "done"),
        ("headache_days", "kept as a number; if missing -> train median + a 'was missing' flag", "done"),
        (
            "Masked copies (train only)",
            f"+1 copy per row with fields blanked at {CFG.headache_model['unknown_augment']:.0%} (chat realism)",
            "done",
        ),
        (
            "Model choice",
            f"Random Forest vs Gradient Boosting, {CFG.headache_model['cv_folds']}-fold stratified CV "
            "on train, best macro-F1 wins",
            "done",
        ),
        ("Test once", "full test split, real patients only, and with 50% of fields unknown", "done"),
        (
            "Inconclusive rule",
            f"top probability < {CFG.safety['low_confidence_threshold']} -> reported as inconclusive, no guess shown",
            "done",
        ),
    ]
    draw_steps(
        "Headache classifier: from table to prediction",
        "Implemented in src/models/headache_clf.py and src/train/train_headache.py",
        steps,
        OUT,
        "07_headache_pipeline_steps",
    )


def main() -> None:
    _, table = seg_tables()
    hd = pd.read_csv(
        CFG.paths["processed"] / "headache_clean.csv",
        keep_default_na=False,
        dtype={c: str for c in H.CATEGORICAL + H.BINARY},
    )
    fig_model_input(table)
    fig_augmentations(table)
    fig_seg_steps(table)
    fig_healthy_scans(table)
    fig_splits(table)
    fig_headache_input(hd)
    fig_unknown_masking(hd)
    fig_headache_steps()
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
