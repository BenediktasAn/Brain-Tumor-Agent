import json
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.config import load_config
from src.data import brain_mri, headache, optional_sources


@dataclass
class Check:
    dataset: str
    name: str
    status: str
    detail: str


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, dataset: str, name: str, ok: bool, detail: str, warn_only: bool = False) -> None:
        status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
        self.checks.append(Check(dataset, name, status, detail))
        print(f"[{status}] {dataset}: {name} - {detail}")

    def skip(self, dataset: str, detail: str) -> None:
        self.checks.append(Check(dataset, "available", "SKIP", detail))
        print(f"[SKIP] {dataset}: {detail}")


def verify_headache(cfg, r: Report) -> None:
    ds = "headache"
    csv = cfg.paths["headache_csv"]
    if not csv.exists():
        return r.skip(ds, f"{csv} missing; see scripts/download_data.md")
    try:
        raw = headache.read_raw(csv)
        r.add(ds, "schema", True, f"{raw.shape[0]:,} rows x {raw.shape[1]} expected columns")
        df = headache.clean(raw)
        r.add(ds, "values", True, "every value is in the allowed set (after cleaning)")
    except headache.DataValidationError as e:
        return r.add(ds, "schema/values", False, str(e))
    counts = df[headache.TARGET].value_counts()
    r.add(ds, "classes", set(counts.index) == set(headache.CLASSES), f"{counts.to_dict()}")
    clean = headache.load_headache(csv, cfg.paths["headache_original_csv"])
    r.add(ds, "duplicates", True, f"{len(df) - len(clean):,} duplicate rows removed -> {len(clean):,} unique")
    feats = clean[headache.FEATURES].apply(tuple, axis=1)
    conflicts = int((clean.groupby(feats)[headache.TARGET].nunique() > 1).sum())
    r.add(ds, "label conflicts", conflicts == 0, f"{conflicts} identical feature rows with different CLASS", True)
    if cfg.paths["headache_original_csv"].exists():
        n_real = int(headache.original_row_mask(raw, cfg.paths["headache_original_csv"]).sum())
        r.add(ds, "real patients", n_real == 849, f"{n_real} of {len(raw):,} raw rows match HeadacheDSS", True)
    else:
        r.add(ds, "real patients", False, "migbase.csv missing; cannot flag real patients", True)


def contact_sheet(rows: list[tuple[Path, Path]], out: Path, threshold: int, tile: int = 256, cols: int = 6) -> None:
    sheet = Image.new("RGB", (tile * cols, tile * ((len(rows) + cols - 1) // cols)))
    for k, (img_p, mask_p) in enumerate(rows):
        img = Image.open(img_p).convert("RGB").resize((tile, tile))
        m = Image.open(mask_p).convert("L").resize((tile, tile), Image.NEAREST).point(lambda v: 255 * (v > threshold))
        img = Image.composite(Image.blend(img, Image.new("RGB", img.size, (208, 59, 59)), 0.45), img, m)
        ImageDraw.Draw(img).text((4, 4), f"{img_p.parent.name}/{img_p.name}", fill=(255, 255, 0))
        sheet.paste(img, ((k % cols) * tile, (k // cols) * tile))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def verify_mri(cfg, r: Report) -> None:
    ds = "mri"
    try:
        items = brain_mri.get_source(cfg.data["mri_source"], cfg.paths["mri_root"]).items()
    except FileNotFoundError as e:
        return r.skip(ds, str(e))
    per_folder = Counter((it.origin, it.label) for it in items)
    empty = [k for k, v in per_folder.items() if v == 0]
    r.add(ds, "class folders", not empty and len(per_folder) >= 7, f"{dict(sorted(per_folder.items()))}")

    seg = [it for it in items if it.origin == "segmentation"]
    no_mask = [it.path.name for it in seg if it.mask_path is None]
    r.add(
        ds,
        "every image has a mask",
        not no_mask,
        f"{len(seg) - len(no_mask)}/{len(seg)} paired; missing: {no_mask[:5]}",
    )
    seg_dirs = {it.path.parent for it in seg}
    orphan = [
        m.name for d in seg_dirs for m in d.glob("*_mask.*") if not m.with_name(m.name.replace("_mask", "")).exists()
    ]
    r.add(ds, "every mask has an image", not orphan, f"orphan masks: {len(orphan)} {orphan[:5]}")

    size_bad, patterns, empty_masks = [], Counter(), 0
    thr = cfg.data["mask_threshold"]
    for it in seg:
        if it.mask_path is None:
            continue
        with Image.open(it.path) as img, Image.open(it.mask_path) as m:
            if img.size != m.size:
                size_bad.append(it.path.name)
            a = np.asarray(m.convert("L"))
        u = tuple(sorted(np.unique(a).tolist()))
        patterns[u if len(u) <= 3 else "other"] += 1
        empty_masks += int((a > thr).sum() == 0)
    r.add(ds, "image/mask same size", not size_bad, f"{len(size_bad)} mismatches {size_bad[:5]}")
    binary = sum(v for k, v in patterns.items() if k != "other" and len(k) == 2)
    r.add(
        ds,
        "mask values",
        True,
        f"{binary}/{len(seg)} strictly two-valued; patterns {dict(patterns.most_common(4))}; binarised at > {thr}",
    )
    r.add(ds, "non-empty masks", empty_masks == 0, f"{empty_masks} masks are empty after thresholding", True)

    random.seed(cfg.seed)
    by_label: dict[str, list] = {}
    for it in seg:
        by_label.setdefault(it.label, []).append(it)
    sample = [(it.path, it.mask_path) for lab in sorted(by_label) for it in random.sample(by_label[lab], 6)]
    out = cfg.paths["figures"] / "verify_mri_contact_sheet.png"
    contact_sheet(sample, out, thr)
    r.add(ds, "contact sheet", True, f"{len(sample)} random overlays -> {out} (inspect by eye)")

    index_path = cfg.paths["processed"] / "mri_index.csv"
    if not index_path.exists():
        r.add(ds, "splits", False, "data/processed/mri_index.csv missing; run scripts/prepare_data.py", True)
        return
    index = pd.read_csv(index_path)
    leaks = brain_mri.check_no_leaks(index)
    r.add(ds, "no duplicate group crosses splits", not leaks, f"{len(leaks)} leaking groups")
    stale = len(index) != len(items)
    r.add(ds, "index up to date", not stale, f"index {len(index):,} rows vs {len(items):,} files", True)
    r.add(
        ds,
        "duplicates",
        True,
        f"{int(index['group'].duplicated(keep=False).sum()):,} images in near-duplicate groups; "
        f"{int(index['label_conflict'].sum())} in label-conflict groups (excluded)",
    )
    r.add(ds, "patient-level leakage", False, "no patient IDs: different slices of one patient may cross splits", True)


def verify_symptom_kb(cfg, r: Report) -> None:
    ds = "symptom_kb"
    if not cfg.features.get("enable_symptom_kb"):
        return r.skip(ds, "disabled in config (features.enable_symptom_kb)")
    try:
        kb = optional_sources.load_symptom_kb(cfg.paths["symptom_kb_dir"])
    except (FileNotFoundError, ValueError) as e:
        return r.add(ds, "load", False, str(e))
    n_dis = kb.patterns["disease"].nunique()
    r.add(
        ds,
        "patterns",
        True,
        f"{len(kb.patterns)} distinct symptom patterns, {n_dis} diseases, {len(kb.symptoms)} symptoms",
    )
    no_desc = sorted(set(kb.patterns["disease"]) - set(kb.descriptions))
    r.add(ds, "descriptions", not no_desc, f"diseases without description: {no_desc}")
    bad_map = sorted(set(optional_sources.HEADACHE_TO_KB.values()) - set(kb.symptoms))
    r.add(ds, "headache -> KB mapping", not bad_map, f"unknown KB symptoms: {bad_map}")
    r.add(ds, "treatment files", True, f"not loaded on purpose; only {', '.join(optional_sources.USED_FILES)} are read")


def main() -> int:
    cfg = load_config()
    r = Report()
    verify_headache(cfg, r)
    verify_mri(cfg, r)
    verify_symptom_kb(cfg, r)
    out = cfg.paths["metrics"] / "verify.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(c) for c in r.checks], indent=2))
    summary = Counter(c.status for c in r.checks)
    print(f"\nsummary: {dict(summary)} -> {out}")
    return 1 if summary["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
