import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from src.data.brain_mri import load_image, load_mask


def standardize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-6)


def build_transform(image_size: int, train: bool) -> v2.Transform:
    resize = v2.Resize((image_size, image_size), antialias=True)
    if not train:
        return resize
    return v2.Compose(
        [
            resize,
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    )


def preprocess(gray01: np.ndarray, image_size: int) -> torch.Tensor:
    x = build_transform(image_size, train=False)(tv_tensors.Image(torch.from_numpy(gray01)[None]))
    return standardize(x.as_subclass(torch.Tensor).float())[None]


class SegDataset(Dataset):
    def __init__(self, table: pd.DataFrame, image_size: int, mask_threshold: int, train: bool):
        self.rows = table.reset_index(drop=True)
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.transform = build_transform(image_size, train)

    def __len__(self) -> int:
        return len(self.rows)

    def raw(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.rows.iloc[i]
        img = load_image(row["path"])
        if row["mask_path"]:
            mask = load_mask(row["mask_path"], self.mask_threshold)
        else:
            mask = np.zeros(img.shape, np.uint8)
        return img, mask

    def __getitem__(self, i: int) -> dict:
        img, mask = self.raw(i)
        image, target = self.transform(
            tv_tensors.Image(torch.from_numpy(img)[None]), tv_tensors.Mask(torch.from_numpy(mask)[None])
        )
        row = self.rows.iloc[i]
        return {
            "image": standardize(image.as_subclass(torch.Tensor).float()),
            "mask": target.as_subclass(torch.Tensor).float(),
            "has_tumor": bool(row["mask_path"]),
            "label": row["label"],
            "path": row["path"],
        }
