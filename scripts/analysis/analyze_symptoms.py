import argparse
import json
from pathlib import Path

import pandas as pd
from plot_style import CRITICAL, MUTED, SLOTS, bar_labels, headline, new_fig, save

DEFAULT_DIR = Path(
    "data/raw/kagglehub/datasets/sibamsamanta07/disease-symptom-prediction-and-treatment-dataset/versions/1"
)
OUT = Path("reports/figures/data/symptoms")
METRICS = Path("reports/metrics/data_symptoms.json")
URGENT = {"Paralysis (brain hemorrhage)", "Heart attack"}


def main(d: Path) -> None:
    t = pd.read_csv(d / "Training.csv")
    t.columns = t.columns.str.strip()
    t["prognosis"] = t["prognosis"].str.strip()
    sym = [c for c in t.columns if c != "prognosis"]
    uniq = t.drop_duplicates()
    per = pd.DataFrame({"rows": t["prognosis"].value_counts(), "unique": uniq["prognosis"].value_counts()})
    with_headache = uniq.loc[uniq["headache"] == 1, "prognosis"].value_counts()
    m = {
        "rows": len(t),
        "unique_rows": len(uniq),
        "diseases": int(t["prognosis"].nunique()),
        "symptoms": len(sym),
        "rows_per_disease": int(per["rows"].iloc[0]),
        "unique_patterns_per_disease": {"min": int(per["unique"].min()), "max": int(per["unique"].max())},
        "diseases_with_headache_pattern": with_headache.to_dict(),
        "files_not_used": ["medications.csv", "diets.csv", "workout_df.csv", "precautions_df.csv"],
    }

    fig, ax = new_fig()
    bars = ax.bar(["rows in Training.csv", "distinct rows"], [len(t), len(uniq)], color=[MUTED, SLOTS[0]], width=0.5)
    bar_labels(ax, bars)
    ax.set_ylabel("rows")
    headline(
        fig,
        f"{len(t):,} rows, but only {len(uniq)} are different",
        f"Each of the {m['diseases']} diseases has {per['unique'].min()}-{per['unique'].max()} symptom patterns, "
        f"each copied until the disease has {m['rows_per_disease']} rows.\nA random train/test split puts copies "
        "of every test row in training -> ~100% accuracy that means nothing. It is a lookup table, not patient data",
    )
    save(fig, OUT, "01_duplicates")

    fig, ax = new_fig()
    s = with_headache.sort_values()
    bars = ax.barh(s.index, s.values, color=[CRITICAL if c in URGENT else SLOTS[0] for c in s.index], height=0.6)
    bar_labels(ax, bars, horizontal=True)
    ax.set_xlabel("distinct symptom patterns that include 'headache'")
    headline(
        fig,
        f"Headache appears in {len(s)} of the {m['diseases']} diseases",
        "Useful for 'other things to discuss with a doctor' (fever + headache -> infections). "
        "Red = conditions that may only ever\nmake the advice MORE urgent, never cancel a scan",
    )
    save(fig, OUT, "02_headache_diseases")

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    main(ap.parse_args().dir)
