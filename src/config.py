from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

SEED = 42

PATHS = {
    "headache_csv": "data/raw/kagglehub/datasets/ashwin0605/headache-diagnosis/versions/1/Headache Based Data.csv",
    "headache_original_csv": "data/raw/headachedss/migbase.csv",
    "mri_root": "data/raw/kagglehub/datasets/indk214/brain-tumor-dataset-segmentation-and-classification/versions/1"
    "/DATASET",
    "symptom_kb_dir": "data/raw/kagglehub/datasets/sibamsamanta07/disease-symptom-prediction-and-treatment-dataset"
    "/versions/1",
    "processed": "data/processed",
    "figures": "reports/figures",
    "metrics": "reports/metrics",
    "models": "models",
    "headache_model": "models/headache/model.joblib",
    "segmentation_model": "models/segmentation/model.pt",
}

DATA = {
    "headache_test_size": 0.2,
    "mri_source": "indk214",
    "mask_threshold": 127,
    "dup_corr_threshold": 0.98,
    "image_size": 256,
    "split": {"train": 0.70, "val": 0.15, "test": 0.15},
    "drop_label_conflicts": True,
    "seg_include_negatives": True,
}

HEADACHE_MODEL = {
    "unknown_augment": 0.15,
    "cv_folds": 5,
}

SEGMENTATION = {
    "encoder": "resnet34",
    "threshold": 0.5,
    "min_area_px": 50,
    "epochs": 10,
    "batch_size": 16,
    "lr": 3e-4,
    "patience": 5,
}

FEATURES = {
    "enable_symptom_kb": True,
    "enable_glioma_module": False,
    "two_dataset_mode": False,
}

SAFETY = {
    "red_flag_threshold": 2,
    "benign_can_cancel_scan": True,
    "low_confidence_threshold": 0.6,
}

LLM = {"model_env": "ANTHROPIC_MODEL", "default_model": "claude-sonnet-5-5"}

OPTIONAL_DATASETS = {"brats_slug": None, "rifat_mri_slug": None}


@dataclass(frozen=True)
class Config:
    seed: int
    paths: dict[str, Path]
    data: dict[str, Any]
    headache_model: dict[str, Any]
    segmentation: dict[str, Any]
    features: dict[str, Any]
    safety: dict[str, Any]
    llm: dict[str, Any]
    optional_datasets: dict[str, Any]


def load_config() -> Config:
    return Config(
        seed=SEED,
        paths={k: (REPO_ROOT / v).resolve() for k, v in PATHS.items()},
        data=dict(DATA),
        headache_model=dict(HEADACHE_MODEL),
        segmentation=dict(SEGMENTATION),
        features=dict(FEATURES),
        safety=dict(SAFETY),
        llm=dict(LLM),
        optional_datasets=dict(OPTIONAL_DATASETS),
    )
