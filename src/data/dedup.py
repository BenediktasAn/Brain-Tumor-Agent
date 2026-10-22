from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image


def thumbnail(img: Image.Image, n: int) -> np.ndarray:
    a = np.asarray(img.convert("L").resize((n, n), Image.BILINEAR), dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def thumbnails(paths: Iterable[Path]) -> tuple[np.ndarray, np.ndarray]:
    t32, t64 = [], []
    for p in paths:
        with Image.open(p) as img:
            t32.append(thumbnail(img, 32))
            t64.append(thumbnail(img, 64))
    return np.stack(t32), np.stack(t64)


def find_duplicate_pairs(
    t32: np.ndarray, t64: np.ndarray, threshold: float = 0.98, block: int = 1024
) -> list[tuple[int, int, float]]:
    a = t32 / np.sqrt(t32.shape[1])
    pairs = []
    for s in range(0, len(a), block):
        ii, jj = np.nonzero(a[s : s + block] @ a.T > threshold - 0.03)
        for i, j in zip(ii + s, jj, strict=True):
            if i < j:
                r = float(t64[i] @ t64[j] / t64.shape[1])
                if r > threshold:
                    pairs.append((int(i), int(j), r))
    return pairs


def group_ids(n: int, pairs: Iterable[tuple[int, int, float]]) -> np.ndarray:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])
