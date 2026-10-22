import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import load_config
from src.data import headache as H
from src.models.headache_clf import HeadacheClassifier, mask_unknowns
from src.train.train_headache import load_processed
from src.utils import write_json


def score(model: HeadacheClassifier, df: pd.DataFrame, low_conf: float) -> dict:
    proba = model.predict_frame(df)
    pred = np.array(model.classes)[proba.argmax(axis=1)]
    y = df[H.TARGET].to_numpy()
    labels = [c for c in H.CLASSES if c in set(y) | set(pred)]
    return {
        "n": len(df),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", labels=sorted(set(y)))), 4),
        "inconclusive_share": round(float((proba.max(axis=1) < low_conf).mean()), 4),
        "per_class": classification_report(y, pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y, pred, labels=labels).tolist()},
    }


def plot_confusion(result: dict, title: str, path: Path) -> None:
    labels, m = result["confusion_matrix"]["labels"], np.array(result["confusion_matrix"]["matrix"])
    share = m / m.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(m):
        ax.text(
            j, i, f"{v}\n{share[i, j]:.0%}", ha="center", va="center", color="white" if share[i, j] > 0.6 else "black"
        )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}\naccuracy {result['accuracy']:.1%}, macro-F1 {result['macro_f1']:.3f} (n={result['n']})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=cfg.paths["headache_model"])
    ap.add_argument("--mask-share", type=float, default=0.5, help="share of fields blanked in the 'short chat' test")
    args = ap.parse_args()

    model = HeadacheClassifier.load(args.model)
    df = load_processed(cfg)
    test = df[df["split"] == "test"].reset_index(drop=True)
    low = cfg.safety["low_confidence_threshold"]
    figs = cfg.paths["figures"] / "models"

    results = {"model": str(args.model), "meta": model.meta, "low_confidence_threshold": low}
    results["test"] = score(model, test, low)
    plot_confusion(results["test"], "Headache type - test split", figs / "headache_confusion_test.png")

    real = test[test["is_original"].astype(str) == "True"]
    if len(real):
        results["test_real_patients"] = score(model, real, low)
        plot_confusion(results["test_real_patients"], "Real patients only", figs / "headache_confusion_real.png")

    masked = mask_unknowns(test, args.mask_share, np.random.default_rng(cfg.seed), model.features)
    results[f"test_{int(args.mask_share * 100)}pct_unknown"] = score(model, masked, low)

    run_dir = cfg.paths["models"] / "headache" / model.meta.get("run", "")
    ablations = {}
    for p in sorted(run_dir.glob("ablation_*.joblib")):
        r = score(HeadacheClassifier.load(p), test, low)
        ablations[p.stem.removeprefix("ablation_")] = {k: r[k] for k in ("accuracy", "macro_f1", "inconclusive_share")}
    results["ablations_test"] = ablations

    out = write_json(cfg.paths["metrics"] / "headache_eval.json", results)
    for k in ("test", "test_real_patients", f"test_{int(args.mask_share * 100)}pct_unknown"):
        if k in results:
            r = results[k]
            print(
                f"{k:24s} n={r['n']:4d} accuracy={r['accuracy']:.3f} macro-F1={r['macro_f1']:.3f} "
                f"inconclusive={r['inconclusive_share']:.1%}"
            )
    for k, r in ablations.items():
        print(f"ablation {k:15s} accuracy={r['accuracy']:.3f} macro-F1={r['macro_f1']:.3f}")
    print(f"-> {out}, figures in {figs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
