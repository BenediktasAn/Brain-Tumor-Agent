import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from plot_style import (
    CRITICAL,
    HEADACHE_COLORS,
    INK_2,
    MUTED,
    SEQ_CMAP,
    SLOTS,
    bar_labels,
    draw_steps,
    headline,
    new_fig,
    save,
    table_fig,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src.data import headache as H

DEFAULT_CSV = Path("data/raw/kagglehub/datasets/ashwin0605/headache-diagnosis/versions/1/Headache Based Data.csv")
DEFAULT_ORIGINAL = Path("data/raw/headachedss/migbase.csv")
OUT = Path("reports/figures/data/headache")
METRICS = Path("reports/metrics/data_headache.json")
CLASSES = ["migraine", "tension", "cluster", "sinus"]
RED_FLAGS = [
    "motor_weakness",
    "speech_disturbance",
    "visual_symptomps",
    "sensory_symptomps",
    "homonymous_symptomps",
    "dysarthria",
    "ataxia",
    "decreased_consciousness",
]
CATEGORICAL = ["durationGroup", "location", "severity", "characterisation", "aura_duration"]
SEED = 42


def heat(ax, table: pd.DataFrame, title: str, fmt: str = "{:.0%}") -> None:
    ax.imshow(table.values, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.grid(False)
    ax.set_title(title, fontsize=12)
    for (i, j), v in np.ndenumerate(table.values):
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9, color="white" if v > 0.55 else "#0b0b0b")
    for s in ax.spines.values():
        s.set_visible(False)


def provenance(df: pd.DataFrame, original_csv: Path) -> dict:
    orig = pd.read_csv(original_csv, dtype=str, keep_default_na=False).apply(lambda s: s.str.strip())
    cols = [c for c in orig.columns if c not in ("headache_days", "previous_attacks")]
    is_orig = df[cols].apply(tuple, axis=1).isin(set(orig[cols].apply(tuple, axis=1)))

    def red(d: pd.DataFrame) -> pd.Series:
        return d[RED_FLAGS].eq("yes").sum(axis=1) >= 2

    out = {
        "original_rows": len(orig),
        "original_class_counts": orig["CLASS"].value_counts().to_dict(),
        "kaggle_rows_matching_original": int(is_orig.sum()),
        "kaggle_rows_generated": int((~is_orig).sum()),
        "generated_by_class": df.loc[~is_orig, "CLASS"].value_counts().to_dict(),
        "ge2_red_flags_original": round(float(red(orig).mean()), 4),
        "ge2_red_flags_generated": round(float(red(df[~is_orig]).mean()), 4),
        "values_only_in_generated_rows": {
            "durationGroup=M": int((df.loc[~is_orig, "durationGroup"] == "M").sum()),
            "location=forehead": int((df.loc[~is_orig, "location"] == "forehead").sum()),
            "characterisation=throbbing": int((df.loc[~is_orig, "characterisation"] == "throbbing").sum()),
        },
        "original_headache_days_values": orig["headache_days"].value_counts().to_dict(),
        "original_previous_attacks_values": orig["previous_attacks"].value_counts().to_dict(),
    }

    fig, (a1, a2) = new_fig(1, 2, gridspec_kw={"width_ratios": [1.5, 1]})
    real = df.loc[is_orig, "CLASS"].value_counts().reindex(CLASSES, fill_value=0)
    gen = df.loc[~is_orig, "CLASS"].value_counts().reindex(CLASSES, fill_value=0)
    a1.barh(CLASSES, real, color=SLOTS[0], height=0.6, label="real patients (original HeadacheDSS / Migbase)")
    a1.barh(CLASSES, gen, left=real, color=MUTED, height=0.6, label="added by the Kaggle uploader")
    for i, c in enumerate(CLASSES):
        a1.annotate(
            f"{real[c]:,} real + {gen[c]:,} added",
            (real[c] + gen[c], i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
        )
    a1.invert_yaxis()
    a1.set_xlim(0, (real + gen).max() * 1.45)
    a1.set_xlabel("rows")
    a1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, fontsize=9.5)
    a1.set_title("Where the 5,150 rows come from", pad=30)
    rates = [out["ge2_red_flags_original"] * 100, out["ge2_red_flags_generated"] * 100]
    bars = a2.bar(["real patients", "added rows"], rates, color=[SLOTS[0], MUTED], width=0.55)
    bar_labels(a2, bars, "{:.1f}%")
    a2.set_ylim(0, 100)
    a2.set_title("Rows with >= 2 red flags", pad=30)
    headline(
        fig,
        f"Only {out['kaggle_rows_matching_original']:,} of {len(df):,} rows are real patients",
        f"The rest ({out['kaggle_rows_generated']:,}) were added, including the whole 'sinus' class. "
        "The 41% red-flag rate and the clean headache_days bands\ncome from the added rows; "
        "in real patients only "
        f"{out['ge2_red_flags_original']:.1%} have >= 2 red flags",
    )
    save(fig, OUT, "00_provenance_real_vs_added")
    return out


DURATION_WORDS = {
    "D": "4-15 min",
    "E": "15-30 min",
    "F": "30 min-3 h",
    "G": "3-4 h",
    "H": "4 h-3 days",
    "I": "3-7 days",
    "J": "> 7 days",
    "M": "undefined code",
}


def example_rows(df: pd.DataFrame, is_orig: pd.Series) -> None:
    rng = np.random.default_rng(SEED)
    binary = [c for c in df.columns if set(df[c].unique()) <= {"yes", "no"}]
    other = [c for c in binary if c not in RED_FLAGS and c not in ("nausea", "photophobia", "phonophobia")]
    picks = []
    for c in CLASSES:
        for real in (True, False):
            pool = df.index[(df["CLASS"] == c) & (is_orig == real)]
            if len(pool):
                picks.append(rng.choice(pool))
        if c == "sinus":
            pool = df.index[(df["CLASS"] == c) & ~df.index.isin(picks)]
            picks.append(rng.choice(pool))
    rows, colours = [], {}
    for i, k in enumerate(picks):
        r = df.loc[k]
        rows.append(
            [
                r["CLASS"],
                "real patient" if is_orig[k] else "generated",
                r["headache_days"],
                f"{r['durationGroup']} ({DURATION_WORDS.get(r['durationGroup'], '?')})",
                r["location"],
                r["severity"],
                r["characterisation"],
                r["nausea"],
                r["photophobia"],
                r["phonophobia"],
                str(int((r[other] == "yes").sum())),
                str(int((r[RED_FLAGS] == "yes").sum())),
            ]
        )
        colours[(i, 0)] = _tint(HEADACHE_COLORS[r["CLASS"]])
    table_fig(
        rows,
        [
            "class",
            "source",
            "days",
            "attack length",
            "location",
            "severity",
            "character",
            "nausea",
            "light hurts",
            "noise hurts",
            f"other yes /{len(other)}",
            "red flags /8",
        ],
        "What the headache data looks like: 8 rows from the file",
        "Picked at random (seed 42), 1 real patient + 1 generated row per class; sinus has no real patients. "
        "The model learns the class from the other columns\n(red flags excluded). Generated rows can carry many "
        "'yes' symptoms and red flags (the generated migraine and cluster rows here); real patients carry few",
        "12_example_rows",
        col_widths=[0.075, 0.095, 0.05, 0.125, 0.08, 0.075, 0.08, 0.065, 0.08, 0.085, 0.095, 0.095],
        out_dir=OUT,
        cell_colors=colours,
        fontsize=9.5,
    )


def _tint(hex_colour: str, alpha: float = 0.25) -> tuple[float, float, float]:
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(1 - alpha + alpha * v for v in (r, g, b))


def barcode(df: pd.DataFrame, is_orig: pd.Series) -> None:
    rng = np.random.default_rng(SEED)
    binary = [c for c in df.columns if set(df[c].unique()) <= {"yes", "no"}]
    cols = [c for c in binary if c not in RED_FLAGS] + RED_FLAGS
    blocks, bounds = [], []
    for c in CLASSES:
        idx = rng.choice(df.index[df["CLASS"] == c], 15, replace=False)
        part = df.loc[idx]
        part = part.assign(_n=(part[cols] == "yes").sum(axis=1)).sort_values("_n")
        blocks.append(part)
        bounds.append(c)
    sample = pd.concat(blocks)
    m = (sample[cols] == "yes").to_numpy(int)
    fig, ax = new_fig(figsize=(13, 9))
    cmap = matplotlib.colors.ListedColormap(["#f0efec", SLOTS[0]])
    ax.imshow(m, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(1, len(CLASSES)):
        ax.axhline(15 * i - 0.5, color="white", lw=4)
    ax.axvline(len(cols) - len(RED_FLAGS) - 0.5, color=CRITICAL, lw=1.5)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=8.5)
    for t in ax.get_xticklabels()[-len(RED_FLAGS) :]:
        t.set_color(CRITICAL)
    ax.set_yticks([15 * i + 7 for i in range(len(CLASSES))], CLASSES, fontsize=12)
    real_pos = np.flatnonzero(is_orig.loc[sample.index].to_numpy())
    ax.scatter(np.full(len(real_pos), -1.2), real_pos, marker="o", s=18, color=INK_2, clip_on=False)
    ax.set_xlim(-1.8, len(cols) - 0.5)
    ax.text(len(cols) - len(RED_FLAGS) / 2 - 0.5, -1.4, "the 8 red flags", ha="center", color=CRITICAL, fontsize=10)
    headline(
        fig,
        "Each row is one patient: blue = 'yes' for that symptom",
        "15 random rows per class, sorted by how many symptoms are 'yes'. Grey dot = real patient (HeadacheDSS); "
        "the rest are generated.\nReal patients have few 'yes' symptoms; the generated 'everything is yes' "
        "rows bring most of the red flags",
    )
    save(fig, OUT, "13_rows_barcode", top=0.88)


def main(csv: Path, original: Path | None = None) -> None:
    raw = pd.read_csv(csv, dtype=str, keep_default_na=False)
    df = raw.apply(lambda s: s.str.strip())
    df["CLASS"] = df["CLASS"].str.lower()
    m: dict = {
        "source_file": str(csv),
        "raw_shape": list(raw.shape),
        "class_counts": df["CLASS"].value_counts().reindex(CLASSES).to_dict(),
    }
    binary = [c for c in df.columns if set(df[c].unique()) <= {"yes", "no"}]
    Y = df[binary].eq("yes")
    rf = Y[RED_FLAGS].sum(axis=1)
    days = df["headache_days"].astype(int)
    colors = [HEADACHE_COLORS[c] for c in CLASSES]

    fig, ax = new_fig()
    counts = df["CLASS"].value_counts().reindex(CLASSES)
    bars = ax.bar(CLASSES, counts.values, color=colors, width=0.6)
    bar_labels(ax, bars)
    ax.set_ylabel("rows")
    headline(
        fig,
        "Four headache classes, moderately imbalanced",
        f"{len(df):,} rows x {df.shape[1]} columns  |  largest/smallest class = {counts.max() / counts.min():.1f}x  "
        "|  no 'brain tumour' class exists in this dataset",
    )
    save(fig, OUT, "01_class_balance")

    fig, axes = new_fig(4, 1, sharex=True)
    rng = {}
    for ax, c in zip(axes, CLASSES):
        d = days[df["CLASS"] == c]
        rng[c] = [int(d.min()), int(d.max())]
        ax.hist(d, bins=np.arange(0.5, 31.5, 1), color=HEADACHE_COLORS[c], rwidth=0.85)
        ax.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=11)
        ax.text(0.995, 0.75, f"range {d.min()}-{d.max()} days", transform=ax.transAxes, ha="right", color=INK_2)
    axes[-1].set_xlabel("headache_days")
    m["headache_days_range_by_class"] = rng
    headline(
        fig,
        "headache_days almost separates the classes on its own",
        "Each class lives in its own band of days - a strong hint the data was generated from rules",
    )
    save(fig, OUT, "02_headache_days_by_class")

    fig, axes = new_fig(1, 5, figsize=(15, 7.5))
    for ax, col in zip(axes, CATEGORICAL):
        t = pd.crosstab(df[col], df["CLASS"], normalize="columns")[CLASSES]
        heat(ax, t, col, "{:.0%}")
        ax.tick_params(axis="x", rotation=45)
    headline(
        fig,
        "Categorical columns: each class has one 'signature' value at ~70%",
        "Cell = % of that class with that value. 0% cells: 'forehead' only in sinus, 'throbbing' only in migraine",
    )
    save(fig, OUT, "03_categorical_by_class")

    rates = Y.groupby(df["CLASS"]).mean().T[CLASSES]
    rates.index = [f"{i}  [RED FLAG]" if i in RED_FLAGS else i for i in rates.index]
    fig, ax = new_fig(figsize=(12, 9))
    heat(ax, rates, "", "{:.0%}")
    headline(
        fig,
        "Yes/no symptoms: most sit at a flat ~20% in every class",
        "Cell = % 'yes' within the class. Flat rows carry no information about the headache type",
    )
    save(fig, OUT, "04_symptom_rates_by_class", top=0.9)

    dist = rf.value_counts().reindex(range(9), fill_value=0)
    fig, ax = new_fig()
    bars = ax.bar(dist.index, dist.values, color=[SLOTS[0] if k < 2 else CRITICAL for k in dist.index], width=0.7)
    bar_labels(ax, bars)
    ax.axvline(1.5, color=INK_2, lw=1)
    ax.text(1.6, dist.max() * 0.9, "scan rule:\n>= 2 red flags", color=INK_2, va="top")
    ax.set_xlabel("number of red flags that are 'yes' (out of 8)")
    ax.set_ylabel("rows")
    ax.set_xticks(range(9))
    n2 = int((rf >= 2).sum())
    m["red_flags"] = {
        "count_distribution": {int(k): int(v) for k, v in dist.items()},
        "rows_ge2": n2,
        "share_ge2": round(n2 / len(df), 4),
        "share_ge2_by_class": (rf >= 2).groupby(df["CLASS"]).mean().round(4).reindex(CLASSES).to_dict(),
    }
    headline(
        fig,
        f"{n2:,} of {len(df):,} rows ({n2 / len(df):.1%}) would trigger the scan rule",
        f"Red flags come in blocks: {dist[0]:,} rows have none, only {dist[1]} have exactly one - "
        "real patients would not look like this",
    )
    save(fig, OUT, "05_red_flag_count_distribution")

    share = (rf >= 2).groupby(df["CLASS"]).mean().reindex(CLASSES)
    fig, ax = new_fig()
    bars = ax.bar(CLASSES, share.values * 100, color=colors, width=0.6)
    bar_labels(ax, bars, "{:.1f}%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of rows with >= 2 red flags")
    headline(
        fig,
        "Red flags are equally common in every headache type",
        f"{share.min():.1%}-{share.max():.1%} in all four classes -> red flags tell us nothing about the type, "
        "so they are kept out of the type classifier",
    )
    save(fig, OUT, "06_red_flag_rate_by_class")

    other = [c for c in binary if c not in RED_FLAGS]
    burden = Y[other].sum(axis=1)
    fig, ax = new_fig()
    bins = np.arange(-0.5, len(other) + 1.5, 1)
    ax.hist(
        burden[rf < 2],
        bins=bins,
        color=SLOTS[0],
        alpha=0.85,
        rwidth=0.85,
        label=f"< 2 red flags (n={int((rf < 2).sum()):,})",
    )
    ax.hist(burden[rf >= 2], bins=bins, color=SLOTS[1], alpha=0.85, rwidth=0.85, label=f">= 2 red flags (n={n2:,})")
    ax.set_xlabel(f"number of OTHER yes/no symptoms that are 'yes' (out of {len(other)})")
    ax.set_ylabel("rows")
    ax.legend(loc="upper right")
    m["other_symptom_mean"] = {
        "lt2_red_flags": round(float(burden[rf < 2].mean()), 2),
        "ge2_red_flags": round(float(burden[rf >= 2].mean()), 2),
    }
    headline(
        fig,
        "Two kinds of rows: 'quiet' ones and 'everything is yes' ones",
        f"Mean other symptoms: {burden[rf < 2].mean():.1f} vs {burden[rf >= 2].mean():.1f}. "
        "Looks like a noise block added by a data generator",
    )
    save(fig, OUT, "07_symptom_burden")

    feats = [
        "nausea",
        "photophobia",
        "phonophobia",
        "lacrimation",
        "conjunctival_injection",
        "rhinorrhoea",
        "ptosis",
        "nasal_congestion",
    ]
    loc_uni = (df["location"] == "unilateral").groupby(df["CLASS"]).mean()
    plaus = Y[feats].groupby(df["CLASS"]).mean().T[CLASSES]
    plaus.loc["location = unilateral"] = loc_uni[CLASSES]
    fig, ax = new_fig()
    x = np.arange(len(plaus))
    w = 0.2
    for k, c in enumerate(CLASSES):
        ax.bar(x + (k - 1.5) * w, plaus[c] * 100, w * 0.9, color=HEADACHE_COLORS[c], label=c)
    ax.set_xticks(x, plaus.index, rotation=25, ha="right")
    ax.set_ylabel("% of class with feature")
    ax.set_ylim(0, 100)
    ax.legend(ncols=4, loc="upper right")
    m["plausibility_rates"] = (plaus * 100).round(1).to_dict()
    eye_nose = plaus.loc[["lacrimation", "conjunctival_injection", "rhinorrhoea", "ptosis"]]
    headline(
        fig,
        "Sanity check against textbook patterns: only partly plausible",
        "Migraine has more nausea and light/sound sensitivity, as expected. But cluster headache (textbook: "
        "strictly one-sided,\nteary red eye, runny nose) is one-sided in only "
        f"{loc_uni['cluster']:.0%} of rows, and its eye/nose symptoms are only slightly raised "
        f"({eye_nose['cluster'].mean():.0%} vs {eye_nose[['migraine', 'tension', 'sinus']].values.mean():.0%})",
    )
    save(fig, OUT, "08_clinical_plausibility")

    pa = pd.to_datetime(df["previous_attacks"], format="%d-%b-%y", errors="coerce")
    monthly = pa.dt.to_period("M").value_counts().sort_index()
    fig, ax = new_fig()
    ax.bar(monthly.index.to_timestamp(), monthly.values, width=25, color=SLOTS[0])
    ax.set_ylabel("rows")
    m["previous_attacks"] = {
        "unparseable": int(pa.isna().sum()),
        "min": str(pa.min().date()),
        "max": str(pa.max().date()),
        "share_2024": round(float((pa.dt.year == 2024).mean()), 4),
    }
    headline(
        fig,
        "previous_attacks holds dates (2019-2024), not a number of attacks",
        f"{(pa.dt.year == 2024).mean():.0%} fall in 2024; no reference date, no difference between classes -> dropped",
    )
    save(fig, OUT, "09_previous_attacks_dates")

    X = df.drop(columns=["CLASS", "previous_attacks"]).apply(lambda s: s.astype("category").cat.codes)
    mi = pd.Series(mutual_info_classif(X, df["CLASS"], discrete_features=True, random_state=SEED), index=X.columns)
    p = df["CLASS"].value_counts(normalize=True)
    h_class = float(-(p * np.log(p)).sum())
    mi = (mi / h_class).sort_values()
    fig, ax = new_fig(figsize=(12, 9))
    labels = [f"{c}  [RED FLAG]" if c in RED_FLAGS else c for c in mi.index]
    bars = ax.barh(labels, mi.values * 100, color=SLOTS[0], height=0.7)
    bar_labels(ax, bars, "{:.1f}%", horizontal=True)
    ax.set_xlabel("% of the class uncertainty explained by this column alone (mutual information / class entropy)")
    ax.tick_params(axis="y", labelsize=9)
    m["mutual_information_pct_of_class_entropy"] = (mi * 100).round(2).sort_values(ascending=False).to_dict()
    headline(
        fig,
        "A handful of columns carry almost all the signal",
        "The 8 [RED FLAG] columns explain ~0% of the headache type - and so do most other symptom columns",
    )
    save(fig, OUT, "10_signal_per_column", top=0.9)

    with_aura_dur = df["aura_duration"].isin(["minutes", "hour", "day"])
    m["consistency"] = {
        "aura_duration_given_but_headache_with_aura_no": int(
            (with_aura_dur & (df["headache_with_aura"] == "no")).sum()
        ),
        "headache_with_aura_yes_but_aura_development_no": int(
            ((df["headache_with_aura"] == "yes") & (df["aura_development"] == "no")).sum()
        ),
        "hemiplegic_yes_but_motor_weakness_no": int(
            ((df["hemiplegic"] == "yes") & (df["motor_weakness"] == "no")).sum()
        ),
        "aura_duration_no_vs_none": df["aura_duration"]
        .isin(["no", "none"])
        .groupby(df["aura_duration"])
        .size()
        .reindex(["no", "none"])
        .to_dict(),
        "all_yes_no_columns_no": int((~Y).all(axis=1).sum()),
    }
    dup = int(df.duplicated().sum())
    m["exact_duplicate_rows"] = dup
    dup_no_dates = int(df.drop(columns="previous_attacks").duplicated().sum())
    m["duplicate_rows_once_previous_attacks_dropped"] = dup_no_dates

    clean_df = H.load_headache(csv, original)
    tr, te = H.split(clean_df, 0.2, SEED)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    acc_days = cross_val_score(
        DecisionTreeClassifier(max_depth=3, random_state=SEED), clean_df[["headache_days"]], clean_df["CLASS"], cv=cv
    ).mean()
    m["separability_check"] = {
        "rows": len(clean_df),
        "majority_class_accuracy": round(float(clean_df["CLASS"].value_counts(normalize=True).max()), 4),
        "depth3_tree_on_headache_days_only_cv5_accuracy": round(float(acc_days), 4),
    }

    n_real = int(clean_df["is_original"].sum()) if clean_df["is_original"].notna().all() else None
    m["preprocessing"] = {
        "after_dedup_rows": len(clean_df),
        "real_patient_rows": n_real,
        "feature_columns": len(H.FEATURES),
        "train_rows": len(tr),
        "test_rows": len(te),
        "real_patient_rows_in_test": int(te["is_original"].sum()) if n_real is not None else None,
    }
    steps = [
        (
            "Load raw CSV",
            f"{raw.shape[0]:,} rows x {raw.shape[1]} columns; strip whitespace, lower-case text "
            "(file uses LF line endings; loader also accepts CRLF)",
            "done",
        ),
        (
            "Validate schema + flag real patients",
            "40 expected columns, 4 classes, allowed values only; "
            "rows matching the original HeadacheDSS data are flagged real",
            "done",
        ),
        ("Drop previous_attacks", "random calendar dates, not a count of attacks (rewritten by the uploader)", "done"),
        (
            "Clean values",
            "aura_duration 'none' -> 'no' (a patient can't tell them apart);  "
            "durationGroup 'M' (undefined code) -> 'unknown'",
            "done",
        ),
        (
            "Remove duplicate rows",
            f"{len(df) - len(clean_df):,} copies removed -> {len(clean_df):,} unique rows "
            f"({n_real} of them real patients). The dates were hiding most copies",
            "done",
        ),
        (
            "Separate red flags from features",
            "8 red-flag columns feed the scan RULE only, never the type classifier "
            f"({len(H.FEATURES)} feature columns remain)",
            "done",
        ),
        (
            "Unknowns are explicit",
            "anything the patient did not say becomes 'unknown' (or empty for headache_days), never a guessed 'no'",
            "done",
        ),
        (
            "Stratified 80/20 split (seed 42)",
            f"train {len(tr):,} / test {len(te):,}; the real patients in test are also scored on their own",
            "done",
        ),
    ]
    draw_steps(
        "Headache data: preprocessing pipeline",
        "Every step is implemented in src/data/headache.py",
        steps,
        OUT,
        "11_preprocessing_steps",
    )

    if original is not None and original.exists():
        m["provenance"] = provenance(df, original)
        is_orig = H.original_row_mask(df, original)
        example_rows(df, is_orig)
        barcode(df, is_orig)

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(m, indent=2, default=str))
    print(
        json.dumps(
            {k: m[k] for k in ["separability_check", "consistency", "preprocessing", "other_symptom_mean"]},
            indent=2,
            default=str,
        )
    )
    print("figures ->", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument(
        "--original", type=Path, default=DEFAULT_ORIGINAL, help="original HeadacheDSS migbase.csv (skipped if missing)"
    )
    args = ap.parse_args()
    main(args.csv, args.original)
