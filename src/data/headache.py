from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "CLASS"
CLASSES = ["migraine", "tension", "cluster", "sinus"]
UNKNOWN = "unknown"

COLUMNS = [
    "CLASS",
    "headache_days",
    "durationGroup",
    "location",
    "severity",
    "characterisation",
    "nausea",
    "vomitting",
    "photophobia",
    "phonophobia",
    "aggravation",
    "pericranial",
    "conjunctival_injection",
    "lacrimation",
    "nasal_congestion",
    "rhinorrhoea",
    "eyelid_oedema",
    "sweating",
    "miosis",
    "ptosis",
    "agitation",
    "motor_weakness",
    "speech_disturbance",
    "visual_symptomps",
    "sensory_symptomps",
    "homonymous_symptomps",
    "dysarthria",
    "vertigo",
    "tinnitus",
    "hypacusia",
    "diplopia",
    "ataxia",
    "decreased_consciousness",
    "nasal_visual_symptomps",
    "paraesthesias",
    "aura_development",
    "headache_with_aura",
    "aura_duration",
    "hemiplegic",
    "previous_attacks",
]
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
ACUTE_SIGNS = ["decreased_consciousness", "motor_weakness", "speech_disturbance"]
DROPPED = ["previous_attacks"]
NUMERIC = ["headache_days"]
CATEGORICAL = ["durationGroup", "location", "severity", "characterisation", "aura_duration"]
BINARY = [c for c in COLUMNS if c not in {TARGET, *NUMERIC, *CATEGORICAL, *DROPPED}]
FEATURES = [c for c in COLUMNS if c not in {TARGET, *DROPPED, *RED_FLAGS}]

DURATION_GROUP_BOUNDS_S: dict[str, tuple[float, float]] = {
    "A": (0, 5),
    "B": (5, 120),
    "C": (120, 240),
    "D": (240, 900),
    "E": (900, 1800),
    "F": (1800, 10800),
    "G": (10800, 14400),
    "H": (14400, 259200),
    "I": (259200, 604800),
    "J": (604800, float("inf")),
}
ALLOWED: dict[str, list[str]] = {
    "durationGroup": [*DURATION_GROUP_BOUNDS_S, UNKNOWN],
    "location": ["unilateral", "bilateral", "orbital", "forehead", UNKNOWN],
    "severity": ["mild", "moderate", "severe", UNKNOWN],
    "characterisation": ["pulsating", "throbbing", "pressing", "stabbing", UNKNOWN],
    "aura_duration": ["no", "minutes", "hour", "day", UNKNOWN],
    **{c: ["yes", "no", UNKNOWN] for c in BINARY},
}
HEADACHE_DAYS_RANGE = (1, 30)


class DataValidationError(ValueError):
    pass


def duration_group(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("duration must be >= 0")
    for code, (lo, hi) in DURATION_GROUP_BOUNDS_S.items():
        if lo <= seconds < hi:
            return code
    raise AssertionError("unreachable: bounds cover [0, inf)")


def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    if list(df.columns) != COLUMNS:
        missing, extra = set(COLUMNS) - set(df.columns), set(df.columns) - set(COLUMNS)
        raise DataValidationError(f"{path.name}: unexpected columns (missing={sorted(missing)}, extra={sorted(extra)})")
    return df.apply(lambda s: s.str.strip())


def original_row_mask(raw: pd.DataFrame, original_csv: Path) -> pd.Series:
    orig = read_raw(original_csv)
    keys = [c for c in COLUMNS if c not in ("headache_days", "previous_attacks")]
    return raw[keys].apply(tuple, axis=1).isin(set(orig[keys].apply(tuple, axis=1)))


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop(columns=DROPPED).copy()
    for c in df.columns:
        if c != "durationGroup":
            df[c] = df[c].str.lower()
    df["durationGroup"] = (
        df["durationGroup"].str.upper().where(lambda s: s.isin(list(DURATION_GROUP_BOUNDS_S)), UNKNOWN)
    )
    df["aura_duration"] = df["aura_duration"].replace({"none": "no"})
    df["headache_days"] = pd.to_numeric(df["headache_days"], errors="raise").astype(int)

    problems = []
    bad_class = ~df[TARGET].isin(CLASSES)
    if bad_class.any():
        problems.append(f"{TARGET}: {sorted(df.loc[bad_class, TARGET].unique())}")
    for c, allowed in ALLOWED.items():
        bad = ~df[c].isin(allowed)
        if bad.any():
            problems.append(f"{c}: {sorted(df.loc[bad, c].unique())[:5]}")
    lo, hi = HEADACHE_DAYS_RANGE
    if not df["headache_days"].between(lo, hi).all():
        problems.append(f"headache_days outside {lo}-{hi}")
    if problems:
        raise DataValidationError("invalid values -> " + "; ".join(problems))
    return df


def load_headache(csv: Path, original_csv: Path | None = None) -> pd.DataFrame:
    raw = read_raw(csv)
    is_orig = original_row_mask(raw, original_csv) if original_csv and original_csv.exists() else None
    df = clean(raw)
    df["is_original"] = is_orig if is_orig is not None else pd.NA
    df = df.loc[~df[[c for c in df.columns if c != "is_original"]].duplicated()].reset_index(drop=True)
    return df


def split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def record_to_features(record: Mapping[str, Any]) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for c in FEATURES:
        v = record.get(c)
        if c in NUMERIC:
            if v is None or v == UNKNOWN:
                row[c] = np.nan
            else:
                days = int(v)
                lo, hi = HEADACHE_DAYS_RANGE
                if not lo <= days <= hi:
                    raise DataValidationError(f"headache_days must be {lo}-{hi}, got {days}")
                row[c] = days
            continue
        if v is None:
            v = UNKNOWN
        elif isinstance(v, bool):
            v = "yes" if v else "no"
        v = str(v).strip()
        v = v.upper() if c == "durationGroup" and v.lower() != UNKNOWN else v.lower()
        if c == "aura_duration" and v == "none":
            v = "no"
        if v not in ALLOWED[c]:
            raise DataValidationError(f"{c}: {v!r} not in {ALLOWED[c]}")
        row[c] = v
    return pd.DataFrame([row], columns=FEATURES)
