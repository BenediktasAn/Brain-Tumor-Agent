from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data import headache as H
from src.data.seg_dataset import SegDataset, preprocess, standardize
from src.models.headache_clf import (
    FEATURE_SETS,
    HeadacheClassifier,
    augment_with_unknowns,
    build_pipeline,
    mask_unknowns,
)
from src.models.segmentation import (
    DiceBCELoss,
    Segmenter,
    build_unet,
    image_metrics,
    overlay,
    remove_small_regions,
    summarize,
)


@pytest.fixture
def toy_headache() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        cls = H.CLASSES[i % 4]
        row = {c: rng.choice(["yes", "no"]) for c in H.BINARY}
        row.update(
            CLASS=cls,
            headache_days=int(rng.integers(1, 31)),
            durationGroup="H",
            location={"migraine": "unilateral", "tension": "bilateral", "cluster": "orbital", "sinus": "forehead"}[cls],
            severity="moderate",
            characterisation="pulsating",
            aura_duration="no",
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_default_features_exclude_red_flags():
    assert not set(FEATURE_SETS["default"]) & set(H.RED_FLAGS)
    assert "headache_days" not in FEATURE_SETS["no_headache_days"]


@pytest.mark.parametrize("kind", ["random_forest", "gradient_boosting"])
def test_pipeline_handles_unknowns_and_roundtrips(kind, toy_headache, tmp_path):
    feats = FEATURE_SETS["default"]
    pipe = build_pipeline(kind, feats, seed=0).fit(toy_headache[feats], toy_headache[H.TARGET])
    clf = HeadacheClassifier(pipe, feats, {"kind": kind})
    pred = clf.predict_record({"location": "orbital"})
    assert pred.label in H.CLASSES
    assert abs(sum(pred.probabilities.values()) - 1) < 1e-3
    assert 0 <= pred.confidence <= 1 and pred.known_fields == 1
    loaded = HeadacheClassifier.load(clf.save(tmp_path / "m.joblib"))
    again = loaded.predict_record({"location": "orbital"})
    assert again.label == pred.label and again.confidence == pytest.approx(pred.confidence, abs=1e-6)


def test_load_missing_model_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_headache"):
        HeadacheClassifier.load(tmp_path / "nope.joblib")


def test_mask_unknowns(toy_headache):
    feats = FEATURE_SETS["default"]
    masked = mask_unknowns(toy_headache, 0.5, np.random.default_rng(0), feats)
    cat = [c for c in feats if c not in H.NUMERIC]
    share = (masked[cat] == H.UNKNOWN).to_numpy().mean()
    assert 0.4 < share < 0.6
    assert masked["headache_days"].isna().any()
    assert (masked[H.TARGET] == toy_headache[H.TARGET]).all()
    assert len(augment_with_unknowns(toy_headache, 0.15, 0, feats)) == 80
    assert augment_with_unknowns(toy_headache, 0, 0, feats) is toy_headache


@pytest.fixture
def seg_table(tmp_path) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i, has_tumour in enumerate([True, False]):
        img = (rng.random((120, 100)) * 255).astype(np.uint8)
        img_path = tmp_path / f"img_{i}.png"
        Image.fromarray(img).save(img_path)
        mask_path = ""
        if has_tumour:
            m = np.full(img.shape, 3, np.uint8)
            m[40:80, 30:60] = 255
            mask_path = str(tmp_path / f"img_{i}_mask.png")
            Image.fromarray(m).save(mask_path)
        rows.append({"path": str(img_path), "mask_path": mask_path, "label": "glioma" if has_tumour else "notumor"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("train", [False, True])
def test_seg_dataset_outputs(seg_table, train):
    torch.manual_seed(0)
    ds = SegDataset(seg_table, image_size=64, mask_threshold=127, train=train)
    s = ds[0]
    assert s["image"].shape == (1, 64, 64) and s["mask"].shape == (1, 64, 64)
    assert set(torch.unique(s["mask"]).tolist()) <= {0.0, 1.0} and s["mask"].sum() > 0
    assert abs(float(s["image"].mean())) < 1e-4 and abs(float(s["image"].std()) - 1) < 1e-2
    assert s["has_tumor"] and not ds[1]["has_tumor"] and ds[1]["mask"].sum() == 0


def test_eval_transform_is_deterministic(seg_table):
    ds = SegDataset(seg_table, image_size=64, mask_threshold=127, train=False)
    assert torch.equal(ds[0]["image"], ds[0]["image"])
    gray = ds.raw(0)[0]
    assert torch.allclose(preprocess(gray, 64)[0], ds[0]["image"], atol=1e-5)


def test_standardize():
    x = standardize(torch.rand(1, 8, 8) * 5 + 3)
    assert abs(float(x.mean())) < 1e-5 and abs(float(x.std()) - 1) < 1e-3


def test_unet_forward_and_loss():
    model = build_unet("resnet34", pretrained=False).eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 1, 64, 64)
    loss = DiceBCELoss()(logits, torch.zeros_like(logits))
    assert torch.isfinite(loss)


def test_image_metrics():
    gt = np.zeros((10, 10), np.uint8)
    gt[2:6, 2:6] = 1
    assert image_metrics(gt, gt)["dice"] == 1.0
    assert image_metrics(np.zeros_like(gt), gt) == {"dice": 0.0, "iou": 0.0, "pred_area": 0, "gt_area": 16}
    assert image_metrics(np.zeros_like(gt), np.zeros_like(gt))["dice"] == 1.0
    half = np.zeros_like(gt)
    half[2:6, 2:4] = 1
    m = image_metrics(half, gt)
    assert m["dice"] == pytest.approx(2 * 8 / 24) and m["iou"] == pytest.approx(0.5)


def test_remove_small_regions():
    m = np.zeros((20, 20), np.uint8)
    m[0:2, 0:2] = 1
    m[10:16, 10:16] = 1
    out = remove_small_regions(m, min_area=10)
    assert out.sum() == 36 and out[0, 0] == 0


def test_summarize_scores_tumour_and_healthy():
    rows = [
        {"dice": 0.8, "iou": 0.7, "pred_area": 10, "gt_area": 12},
        {"dice": 0.6, "iou": 0.5, "pred_area": 0, "gt_area": 12},
        {"dice": 1.0, "iou": 1.0, "pred_area": 0, "gt_area": 0},
        {"dice": 0.0, "iou": 0.0, "pred_area": 5, "gt_area": 0},
    ]
    s = summarize(rows)
    assert s["dice_mean"] == 0.7 and s["missed_tumours"] == 1
    assert s["false_alarm_rate"] == 0.5 and s["val_score"] == pytest.approx(0.6)


def test_segmenter_predicts_at_original_resolution(tmp_path, seg_table):
    cfg = {"encoder": "resnet34", "image_size": 64, "threshold": 0.5, "min_area_px": 0}
    ckpt = tmp_path / "model.pt"
    torch.save({"state_dict": build_unet("resnet34", pretrained=False).state_dict(), "config": cfg}, ckpt)
    res = Segmenter.load(ckpt).predict(Path(seg_table.loc[0, "path"]))
    assert res.mask.shape == (120, 100) == res.image_shape
    assert res.area_px == int(res.mask.sum())
    assert res.area_pct == pytest.approx(round(100 * res.area_px / 12000, 3))
    assert overlay(np.zeros((120, 100), np.float32), res.mask).shape == (120, 100, 3)


def test_segmenter_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_seg"):
        Segmenter.load(tmp_path / "missing.pt")
