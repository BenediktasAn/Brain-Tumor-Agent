import argparse
import json
import sys
import time

import pandas as pd

from src.config import load_config
from src.data import brain_mri, headache


def prepare_headache(cfg) -> dict:
    raw_rows = len(headache.read_raw(cfg.paths["headache_csv"]))
    df = headache.load_headache(cfg.paths["headache_csv"], cfg.paths["headache_original_csv"])
    train, test = headache.split(df, cfg.data["headache_test_size"], cfg.seed)
    out = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    path = cfg.paths["processed"] / "headache_clean.csv"
    out.to_csv(path, index=False)
    real_known = out["is_original"].notna().all()
    return {
        "file": str(path.relative_to(cfg.paths["processed"].parents[1])),
        "raw_rows": raw_rows,
        "unique_rows": len(df),
        "real_patient_rows": int(out["is_original"].sum()) if real_known else None,
        "by_split": out.groupby("split")["CLASS"].value_counts().unstack().to_dict("index"),
        "real_patients_in_test": int(out.loc[out["split"] == "test", "is_original"].sum()) if real_known else None,
    }


def prepare_mri(cfg) -> dict:
    t0 = time.time()
    source = brain_mri.get_source(cfg.data["mri_source"], cfg.paths["mri_root"])
    index, stats = brain_mri.build_index(
        source, cfg.data["dup_corr_threshold"], cfg.data["split"], cfg.seed, cfg.data["drop_label_conflicts"]
    )
    leaks = brain_mri.check_no_leaks(index)
    if leaks:
        raise RuntimeError(f"{len(leaks)} duplicate groups span several splits; split assignment is broken")
    path = cfg.paths["processed"] / "mri_index.csv"
    index.to_csv(path, index=False)
    seg = brain_mri.segmentation_table(index, cfg.data["seg_include_negatives"])
    cls = brain_mri.classification_table(index)

    def per_split(t: pd.DataFrame) -> dict:
        return t.groupby("split")["label"].value_counts().unstack(fill_value=0).to_dict("index")

    return {
        "file": str(path.relative_to(cfg.paths["processed"].parents[1])),
        "source": source.name,
        **stats,
        "segmentation_rows_by_split": per_split(seg),
        "segmentation_negatives_included": cfg.data["seg_include_negatives"],
        "classification_rows_by_split": per_split(cls),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["headache", "mri"])
    args = ap.parse_args()
    cfg = load_config()
    cfg.paths["processed"].mkdir(parents=True, exist_ok=True)
    report_path = cfg.paths["metrics"] / "prepare_data.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for name, fn in [("headache", prepare_headache), ("mri", prepare_mri)]:
        if args.only and args.only != name:
            continue
        try:
            report[name] = fn(cfg)
            print(f"[ok] {name}: {json.dumps(report[name], default=str)[:300]}...")
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}", file=sys.stderr)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print("report ->", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
