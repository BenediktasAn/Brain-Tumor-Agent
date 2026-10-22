import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from PIL import Image

from src.data.dedup import find_duplicate_pairs, group_ids, thumbnails

LABELS = ["glioma", "meningioma", "notumor", "pituitary"]
SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def canonical_label(folder_name: str) -> str:
    s = re.sub(r"[^a-z]", "", folder_name.lower())
    for label in LABELS:
        if s.startswith(label[:5]):
            return label
    if s in {"no", "notumour", "healthy", "normal"}:
        return "notumor"
    raise ValueError(f"cannot map folder {folder_name!r} to one of {LABELS}")


@dataclass(frozen=True)
class MriItem:
    path: Path
    label: str
    origin: str
    mask_path: Path | None = None


class MriSource(Protocol):
    name: str

    def items(self) -> list[MriItem]: ...


def _child(parent: Path, name: str) -> Path:
    for p in parent.iterdir():
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    raise FileNotFoundError(f"no folder {name!r} in {parent}")


class Indk214Source:
    name = "indk214"

    def __init__(self, root: Path):
        self.root = Path(root)

    def items(self) -> list[MriItem]:
        if not self.root.exists():
            raise FileNotFoundError(f"MRI dataset not found at {self.root}; see scripts/download_data.md")
        out: list[MriItem] = []
        for class_dir in sorted(p for p in _child(self.root, "Segmentation").iterdir() if p.is_dir()):
            label = canonical_label(class_dir.name)
            for img in sorted(class_dir.iterdir()):
                if img.suffix.lower() in IMAGE_EXTS and not img.stem.endswith("_mask"):
                    mask = img.with_name(f"{img.stem}_mask{img.suffix}")
                    out.append(MriItem(img, label, "segmentation", mask if mask.exists() else None))
        cls_root = _child(self.root, "classification")
        for split_dir in sorted(p for p in cls_root.iterdir() if p.is_dir()):
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                label = canonical_label(class_dir.name)
                out += [
                    MriItem(p, label, f"classification/{split_dir.name}")
                    for p in sorted(class_dir.iterdir())
                    if p.suffix.lower() in IMAGE_EXTS
                ]
        return out


SOURCES: dict[str, type] = {"indk214": Indk214Source}


def get_source(name: str, root: Path) -> MriSource:
    if name not in SOURCES:
        raise ValueError(f"unknown MRI source {name!r}; available: {sorted(SOURCES)}")
    return SOURCES[name](root)


def load_image(path: Path, size: int | None = None) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        if size:
            img = img.resize((size, size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def load_mask(path: Path, threshold: int, size: int | None = None) -> np.ndarray:
    with Image.open(path) as m:
        m = m.convert("L")
        if size:
            m = m.resize((size, size), Image.NEAREST)
        return (np.asarray(m) > threshold).astype(np.uint8)


def assign_splits(groups: np.ndarray, labels: np.ndarray, fractions: dict[str, float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(len(groups), dtype=object)
    group_label = pd.Series(labels).groupby(groups).agg(lambda s: Counter(s).most_common(1)[0][0])
    group_size = pd.Series(groups).value_counts()
    bounds = np.cumsum([fractions[s] for s in SPLITS])
    for label in sorted(set(group_label)):
        gs = np.array(sorted(group_label.index[group_label == label]))
        rng.shuffle(gs)
        total = group_size[gs].sum()
        filled = 0
        for g in gs:
            mid = (filled + group_size[g] / 2) / total
            out[groups == g] = SPLITS[min(int(np.searchsorted(bounds, mid)), 2)]
            filled += group_size[g]
    return out


def build_index(
    source: MriSource, dup_threshold: float, fractions: dict[str, float], seed: int, drop_label_conflicts: bool = True
) -> tuple[pd.DataFrame, dict]:
    items = source.items()
    t32, t64 = thumbnails(it.path for it in items)
    pairs = find_duplicate_pairs(t32, t64, dup_threshold)
    groups = group_ids(len(items), pairs)
    df = pd.DataFrame(
        {
            "path": [str(it.path) for it in items],
            "label": [it.label for it in items],
            "origin": [it.origin for it in items],
            "mask_path": [str(it.mask_path) if it.mask_path else "" for it in items],
            "group": groups,
        }
    )
    df["label_conflict"] = df.groupby("group")["label"].transform("nunique") > 1
    keep = ~df["label_conflict"] if drop_label_conflicts else np.ones(len(df), bool)
    df["split"] = "excluded"
    df.loc[keep, "split"] = assign_splits(
        df.loc[keep, "group"].to_numpy(), df.loc[keep, "label"].to_numpy(), fractions, seed
    )
    stats = {
        "images": len(df),
        "duplicate_pairs": len(pairs),
        "duplicate_groups": int((df["group"].value_counts() > 1).sum()),
        "images_in_duplicate_groups": int(df["group"].duplicated(keep=False).sum()),
        "label_conflict_images": int(df["label_conflict"].sum()),
        "excluded": int((df["split"] == "excluded").sum()),
    }
    return df, stats


def segmentation_table(index: pd.DataFrame, include_negatives: bool) -> pd.DataFrame:
    usable = index[index["split"] != "excluded"]
    pos = usable[(usable["origin"] == "segmentation") & (usable["mask_path"] != "")]
    if not include_negatives:
        return pos.reset_index(drop=True)
    neg = usable[(usable["label"] == "notumor") & usable["origin"].str.startswith("classification")]
    return pd.concat([pos, neg], ignore_index=True)


def classification_table(index: pd.DataFrame) -> pd.DataFrame:
    usable = index[(index["split"] != "excluded") & index["origin"].str.startswith("classification")]
    return usable.reset_index(drop=True)


def check_no_leaks(index: pd.DataFrame) -> list[int]:
    used = index[index["split"] != "excluded"]
    n = used.groupby("group")["split"].nunique()
    return n.index[n > 1].tolist()
