import argparse
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.config import load_config
from src.data import headache as H
from src.models.headache_clf import (
    FEATURE_SETS,
    MODEL_KINDS,
    HeadacheClassifier,
    augment_with_unknowns,
    build_pipeline,
)
from src.utils import run_name, set_seed, write_json


def load_processed(cfg) -> pd.DataFrame:
    path = cfg.paths["processed"] / "headache_clean.csv"
    if not path.exists():
        sys.exit(f"{path} missing - run: python scripts/prepare_data.py --only headache")
    return pd.read_csv(path, keep_default_na=False, dtype={c: str for c in H.CATEGORICAL + H.BINARY})


def cross_validate(
    train: pd.DataFrame, kind: str, features: list[str], folds: int, augment: float, seed: int, rounds: int | None
) -> dict:
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    acc, f1 = [], []
    for k, (tr, va) in enumerate(skf.split(train, train[H.TARGET])):
        fit_df = augment_with_unknowns(train.iloc[tr], augment, seed + k, features)
        model = build_pipeline(kind, features, seed, rounds).fit(fit_df[features], fit_df[H.TARGET])
        pred = model.predict(train.iloc[va][features])
        acc.append(accuracy_score(train.iloc[va][H.TARGET], pred))
        f1.append(f1_score(train.iloc[va][H.TARGET], pred, average="macro"))
    return {
        "accuracy_mean": round(float(np.mean(acc)), 4),
        "accuracy_std": round(float(np.std(acc)), 4),
        "macro_f1_mean": round(float(np.mean(f1)), 4),
        "macro_f1_std": round(float(np.std(f1)), 4),
    }


def fit(train: pd.DataFrame, kind: str, features: list[str], augment: float, seed: int, rounds: int | None, meta: dict):
    fit_df = augment_with_unknowns(train, augment, seed, features)
    pipe = build_pipeline(kind, features, seed, rounds).fit(fit_df[features], fit_df[H.TARGET])
    return HeadacheClassifier(pipe, features, meta)


def main() -> int:
    cfg = load_config()
    hp = cfg.headache_model
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=[*MODEL_KINDS, "both"], default="both")
    ap.add_argument("--unknown-augment", type=float, default=hp.get("unknown_augment", 0.15))
    ap.add_argument("--cv-folds", type=int, default=hp.get("cv_folds", 5))
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--rounds", type=int, default=None, help="boosting rounds / forest trees (default 300 / 400)")
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="load data and build pipelines, then stop (no fitting)")
    args = ap.parse_args()

    set_seed(args.seed)
    df = load_processed(cfg)
    train = df[df["split"] == "train"].reset_index(drop=True)
    kinds = list(MODEL_KINDS) if args.model == "both" else [args.model]
    print(f"train rows: {len(train)} {train[H.TARGET].value_counts().to_dict()} | models: {kinds}")
    if args.dry_run:
        for kind in kinds:
            for name, feats in FEATURE_SETS.items():
                build_pipeline(kind, feats, args.seed, args.rounds)
                missing = set(feats) - set(train.columns)
                assert not missing, f"{name}: columns missing from data: {missing}"
        print("dry run OK: data and pipelines are valid; nothing was trained")
        return 0

    run = args.run_name or run_name("headache")
    out = cfg.paths["models"] / "headache" / run
    config = {**vars(args), "run": run, "features": FEATURE_SETS["default"], "train_rows": len(train)}
    write_json(out / "config.json", config)

    cv = {}
    for kind in kinds:
        cv[kind] = cross_validate(
            train, kind, FEATURE_SETS["default"], args.cv_folds, args.unknown_augment, args.seed, args.rounds
        )
        print(f"CV {kind}: {cv[kind]}")
    best = max(kinds, key=lambda k: cv[k]["macro_f1_mean"])
    print(f"chosen: {best}")

    meta = {
        "run": run,
        "kind": best,
        "rounds": args.rounds,
        "feature_set": "default",
        "unknown_augment": args.unknown_augment,
        "cv": cv[best],
    }
    model = fit(train, best, FEATURE_SETS["default"], args.unknown_augment, args.seed, args.rounds, meta)
    model.save(out / "model.joblib")

    ablations = {}
    if not args.skip_ablations:
        for name, feats in FEATURE_SETS.items():
            if name == "default":
                continue
            ablations[name] = cross_validate(
                train, best, feats, args.cv_folds, args.unknown_augment, args.seed, args.rounds
            )
            print(f"CV ablation {name}: {ablations[name]}")
            fit(train, best, feats, args.unknown_augment, args.seed, args.rounds, {**meta, "feature_set": name}).save(
                out / f"ablation_{name}.joblib"
            )

    write_json(out / "cv_metrics.json", {"cv": cv, "chosen": best, "ablations_cv": ablations})
    shutil.copy2(out / "model.joblib", cfg.paths["headache_model"])
    print(f"saved {out} and copied the model to {cfg.paths['headache_model']}")
    print("next: python -m src.eval.evaluate_headache")
    return 0


if __name__ == "__main__":
    sys.exit(main())
