import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.config import load_config
from src.data import brain_mri, headache, optional_sources
from src.data.dedup import find_duplicate_pairs, group_ids, thumbnails

CFG = load_config()


def raw_row(**overrides: str) -> dict[str, str]:
    row = {c: "no" for c in headache.COLUMNS}
    row.update(
        CLASS="migraine",
        headache_days="3",
        durationGroup="H",
        location="unilateral",
        severity="moderate",
        characterisation="pulsating",
        aura_duration="none",
        previous_attacks="02-Feb-22",
    )
    row.update(overrides)
    return row


def write_csv(path: Path, rows: list[dict], line_ending: str = "\n") -> Path:
    lines = [",".join(headache.COLUMNS)] + [",".join(r[c] for c in headache.COLUMNS) for r in rows]
    path.write_bytes((line_ending.join(lines) + line_ending).encode())
    return path


def test_read_raw_accepts_crlf(tmp_path):
    df = headache.read_raw(write_csv(tmp_path / "h.csv", [raw_row(), raw_row(CLASS="tension")], "\r\n"))
    assert df.shape == (2, 40)
    assert df["previous_attacks"].iloc[0] == "02-Feb-22"


def test_read_raw_rejects_wrong_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("CLASS,foo\nmigraine,1\n")
    with pytest.raises(headache.DataValidationError, match="unexpected columns"):
        headache.read_raw(p)


def test_clean_normalises_values():
    raw = pd.DataFrame([raw_row(durationGroup="M", aura_duration="none", location="Unilateral", nausea="YES")])
    df = headache.clean(raw)
    assert "previous_attacks" not in df.columns
    row = df.iloc[0]
    assert row["durationGroup"] == headache.UNKNOWN
    assert row["aura_duration"] == "no"
    assert row["location"] == "unilateral"
    assert row["nausea"] == "yes"


@pytest.mark.parametrize("col,value", [("severity", "extreme"), ("CLASS", "brain_tumour"), ("headache_days", "45")])
def test_clean_rejects_invalid_values(col, value):
    with pytest.raises(headache.DataValidationError):
        headache.clean(pd.DataFrame([raw_row(**{col: value})]))


def test_load_headache_dedupes_rows_that_differ_only_by_date_and_flags_real_rows(tmp_path):
    rows = [raw_row(previous_attacks="01-Jan-24"), raw_row(previous_attacks="05-May-23"), raw_row(CLASS="tension")]
    kaggle = write_csv(tmp_path / "kaggle.csv", rows)
    original = write_csv(tmp_path / "orig.csv", [raw_row(CLASS="tension", headache_days="7", previous_attacks="20")])
    df = headache.load_headache(kaggle, original)
    assert len(df) == 2
    assert df.set_index("CLASS")["is_original"].to_dict() == {"migraine": False, "tension": True}


def test_red_flags_are_never_classifier_features():
    assert not set(headache.RED_FLAGS) & set(headache.FEATURES)
    assert set(headache.ACUTE_SIGNS) <= set(headache.RED_FLAGS)
    assert "previous_attacks" not in headache.FEATURES
    assert len(headache.RED_FLAGS) == 8


@pytest.mark.parametrize(
    "seconds,code",
    [
        (0, "A"),
        (4.9, "A"),
        (5, "B"),
        (10 * 60, "D"),
        (20 * 60, "E"),
        (2 * 3600, "F"),
        (3.5 * 3600, "G"),
        (4 * 3600, "H"),
        (72 * 3600 - 1, "H"),
        (72 * 3600, "I"),
        (30 * 86400, "J"),
    ],
)
def test_duration_group(seconds, code):
    assert headache.duration_group(seconds) == code


def test_duration_group_rejects_negative():
    with pytest.raises(ValueError):
        headache.duration_group(-1)


def test_record_to_features_fills_unknowns_explicitly():
    x = headache.record_to_features(
        {"nausea": True, "photophobia": False, "location": "Orbital", "aura_duration": "none"}
    )
    row = x.iloc[0]
    assert list(x.columns) == headache.FEATURES
    assert row["nausea"] == "yes" and row["photophobia"] == "no" and row["location"] == "orbital"
    assert row["aura_duration"] == "no"
    assert row["severity"] == headache.UNKNOWN and row["vomitting"] == headache.UNKNOWN
    assert math.isnan(row["headache_days"])


def test_record_to_features_ignores_red_flags_and_validates():
    x = headache.record_to_features({"motor_weakness": "yes", "durationGroup": "h", "headache_days": 30})
    assert "motor_weakness" not in x.columns
    assert x.iloc[0]["durationGroup"] == "H"
    with pytest.raises(headache.DataValidationError):
        headache.record_to_features({"location": "left knee"})
    with pytest.raises(headache.DataValidationError):
        headache.record_to_features({"headache_days": 0})


def test_split_is_stratified_and_disjoint():
    df = pd.DataFrame({"CLASS": ["migraine"] * 60 + ["sinus"] * 40, "x": range(100)})
    train, test = headache.split(df, 0.2, seed=0)
    assert len(test) == 20 and (test["CLASS"] == "sinus").sum() == 8
    assert not set(train["x"]) & set(test["x"])


def noise_image(seed: int, size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((size, size)) * 255).astype(np.uint8)


def save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture
def mri_root(tmp_path) -> Path:
    root = tmp_path / "DATASET"
    seed = 0
    for folder in ["Glioma", "Pituitary tumor"]:
        for _ in range(3):
            seed += 1
            img = noise_image(seed)
            mask = np.full(img.shape, 3, np.uint8)
            mask[30:50, 30:50] = 255
            save(img, root / "Segmentation" / folder / f"enh_{seed}.png")
            save(mask, root / "Segmentation" / folder / f"enh_{seed}_mask.png")
    for split in ["Training", "Testing"]:
        for label in ["glioma", "meningioma", "notumor", "pituitary"]:
            for _ in range(3):
                seed += 1
                save(noise_image(seed), root / "classification" / split / label / f"img_{seed}.jpg")
    dup = noise_image(999)
    save(dup, root / "classification" / "Training" / "notumor" / "dup_a.jpg")
    save(
        np.clip(dup.astype(int) * 0.8 + 20, 0, 255).astype(np.uint8), root / "classification/Testing/notumor/dup_b.jpg"
    )
    Image.open(root / "Segmentation" / "Glioma" / "enh_1.png").save(root / "classification/Training/meningioma/c.png")
    return root


@pytest.mark.parametrize(
    "name,label",
    [
        ("Glioma", "glioma"),
        ("Pituitary tumor", "pituitary"),
        ("notumor", "notumor"),
        ("No Tumor", "notumor"),
        ("no_tumor", "notumor"),
        ("meningioma_tumor", "meningioma"),
    ],
)
def test_canonical_label(name, label):
    assert brain_mri.canonical_label(name) == label


def test_canonical_label_rejects_unknown():
    with pytest.raises(ValueError):
        brain_mri.canonical_label("lymphoma")


def test_source_discovers_layout(mri_root):
    items = brain_mri.get_source("indk214", mri_root).items()
    seg = [it for it in items if it.origin == "segmentation"]
    assert len(seg) == 6 and all(it.mask_path is not None for it in seg)
    assert {it.label for it in seg} == {"glioma", "pituitary"}
    assert len(items) == 6 + 24 + 3


def test_source_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        brain_mri.get_source("indk214", tmp_path / "missing").items()
    with pytest.raises(ValueError):
        brain_mri.get_source("nope", tmp_path)


def test_load_mask_binarises_background_of_3(mri_root):
    mask_path = mri_root / "Segmentation" / "Glioma" / "enh_1_mask.png"
    m = brain_mri.load_mask(mask_path, threshold=127)
    assert set(np.unique(m)) == {0, 1}
    assert m.sum() == 20 * 20
    resized = brain_mri.load_mask(mask_path, threshold=127, size=48)
    assert set(np.unique(resized)) == {0, 1}
    img = brain_mri.load_image(mri_root / "Segmentation" / "Glioma" / "enh_1.png", size=48)
    assert img.shape == (48, 48) and img.dtype == np.float32 and 0 <= img.min() and img.max() <= 1


def test_duplicate_detection_is_brightness_invariant_and_ignores_noise(mri_root):
    paths = [
        mri_root / "classification/Training/notumor/dup_a.jpg",
        mri_root / "classification/Testing/notumor/dup_b.jpg",
    ]
    paths += sorted((mri_root / "classification/Training/glioma").glob("*.jpg"))
    t32, t64 = thumbnails(paths)
    pairs = find_duplicate_pairs(t32, t64, 0.98)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    assert list(group_ids(len(paths), pairs)) == [0, 0, 2, 3, 4]


def test_build_index_groups_duplicates_and_excludes_conflicts(mri_root):
    source = brain_mri.get_source("indk214", mri_root)
    index, stats = brain_mri.build_index(source, 0.98, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    dup = index[index["path"].str.contains("dup_")]
    assert dup["group"].nunique() == 1 and dup["split"].nunique() == 1
    conflict = index[index["label_conflict"]]
    assert len(conflict) == 2 and set(conflict["split"]) == {"excluded"}
    assert stats["label_conflict_images"] == 2
    assert brain_mri.check_no_leaks(index) == []


def test_segmentation_and_classification_tables(mri_root):
    source = brain_mri.get_source("indk214", mri_root)
    index, _ = brain_mri.build_index(source, 0.98, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    with_neg = brain_mri.segmentation_table(index, include_negatives=True)
    without = brain_mri.segmentation_table(index, include_negatives=False)
    assert (without["mask_path"] != "").all() and len(without) == 5
    negatives = with_neg[with_neg["mask_path"] == ""]
    assert len(negatives) == 8 and set(negatives["label"]) == {"notumor"}
    cls = brain_mri.classification_table(index)
    assert cls["origin"].str.startswith("classification").all() and "excluded" not in set(cls["split"])


def test_assign_splits_fractions_and_group_integrity():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(1000), rng.integers(1, 4, 1000))
    labels = np.where(groups % 3 == 0, "glioma", "notumor")
    split = brain_mri.assign_splits(groups, labels, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    df = pd.DataFrame({"g": groups, "label": labels, "split": split})
    assert (df.groupby("g")["split"].nunique() == 1).all()
    for _, part in df.groupby("label"):
        share = part["split"].value_counts(normalize=True)
        assert abs(share["train"] - 0.7) < 0.03 and abs(share["test"] - 0.15) < 0.03


def test_assign_splits_is_deterministic():
    groups = np.arange(200)
    labels = np.array(["glioma"] * 200)
    fr = {"train": 0.7, "val": 0.15, "test": 0.15}
    assert (brain_mri.assign_splits(groups, labels, fr, 5) == brain_mri.assign_splits(groups, labels, fr, 5)).all()


@pytest.fixture
def kb_dir(tmp_path) -> Path:
    cols = [
        "headache",
        "high_fever",
        "chills",
        "slurred_speech",
        "weakness_of_one_body_side",
        "spotting_ urination",
        "fluid_overload",
        "fluid_overload",
    ]
    rows = [
        ([1, 1, 1, 0, 0, 0, 0, 0], "Malaria"),
        ([1, 1, 1, 0, 0, 0, 0, 0], "Malaria"),
        ([1, 0, 0, 1, 1, 0, 0, 0], "Paralysis (brain hemorrhage)"),
        ([0, 0, 0, 0, 0, 1, 0, 1], "Peptic ulcer diseae"),
    ]
    lines = [",".join([*cols, "prognosis"])] + [",".join(map(str, v)) + f",{d} " for v, d in rows]
    (tmp_path / "Training.csv").write_text("\n".join(lines) + "\n")
    sev = [
        "Symptom,weight",
        "headache,3",
        "high_fever,7",
        "chills,3",
        "slurred_speech,4",
        "weakness_of_one_body_side,4",
        "spotting_ urination,6",
        "fluid_overload,6",
        "fluid_overload,6",
        "prognosis,5",
    ]
    (tmp_path / "Symptom-severity.csv").write_text("\n".join(sev) + "\n")
    (tmp_path / "description.csv").write_text(
        "Disease,Description\nMalaria,Fever from parasites.\nParalysis (brain hemorrhage),Bleeding in the brain.\n"
        "Peptic ulcer disease,Sores in the stomach lining.\n"
    )
    (tmp_path / "medications.csv").write_text("this file must never be read\n")
    return tmp_path


def test_kb_cleans_names_and_dedupes(kb_dir):
    kb = optional_sources.load_symptom_kb(kb_dir)
    assert len(kb.patterns) == 3
    assert "spotting_urination" in kb.symptoms and "fluid_overload" in kb.symptoms
    assert len(kb.symptoms) == 7
    ulcer = kb.patterns[kb.patterns["disease"] == "Peptic ulcer disease"].iloc[0]
    assert ulcer["fluid_overload"] == 1
    assert kb.descriptions["Peptic ulcer disease"].startswith("Sores")


def test_kb_never_reads_treatment_files(kb_dir, monkeypatch):
    read = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda p, *a, **k: read.append(Path(p).name) or real(p, *a, **k))
    optional_sources.load_symptom_kb(kb_dir)
    assert set(read) == set(optional_sources.USED_FILES)


def test_related_conditions_ranks_and_flags_urgent(kb_dir):
    kb = optional_sources.load_symptom_kb(kb_dir)
    fever = kb.related_conditions({"headache", "high_fever", "chills"})
    assert fever[0].disease == "Malaria" and fever[0].score == 1.0 and not fever[0].urgent
    stroke_like = kb.related_conditions({"headache", "slurred_speech", "weakness_of_one_body_side"})
    assert stroke_like[0].disease == "Paralysis (brain hemorrhage)" and stroke_like[0].urgent


def test_related_conditions_needs_enough_overlap(kb_dir):
    kb = optional_sources.load_symptom_kb(kb_dir)
    assert kb.related_conditions({"headache"}) == []
    assert kb.related_conditions({"headache", "made_up_symptom"}) == []


def test_headache_record_to_kb_symptoms():
    s = optional_sources.headache_record_to_kb_symptoms(
        {"vomitting": "yes", "nausea": "no", "speech_disturbance": "YES"}
    )
    assert s == {"headache", "vomiting", "slurred_speech"}


def test_kb_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        optional_sources.load_symptom_kb(tmp_path)


needs = {
    "headache": pytest.mark.skipif(not CFG.paths["headache_csv"].exists(), reason="headache CSV not downloaded"),
    "original": pytest.mark.skipif(not CFG.paths["headache_original_csv"].exists(), reason="migbase.csv missing"),
    "mri_index": pytest.mark.skipif(
        not (CFG.paths["processed"] / "mri_index.csv").exists(), reason="run scripts/prepare_data.py"
    ),
    "kb": pytest.mark.skipif(not CFG.paths["symptom_kb_dir"].exists(), reason="symptom dataset not downloaded"),
}


@pytest.mark.realdata
@needs["headache"]
@needs["original"]
def test_real_headache_counts():
    raw = headache.read_raw(CFG.paths["headache_csv"])
    assert raw.shape == (5150, 40)
    assert int(headache.original_row_mask(raw, CFG.paths["headache_original_csv"]).sum()) == 849
    df = headache.load_headache(CFG.paths["headache_csv"], CFG.paths["headache_original_csv"])
    assert len(df) == 2748 and int(df["is_original"].sum()) == 540


@pytest.mark.realdata
@needs["mri_index"]
def test_real_mri_index_has_no_leaks():
    index = pd.read_csv(CFG.paths["processed"] / "mri_index.csv")
    assert len(index) == 9215
    assert brain_mri.check_no_leaks(index) == []
    seg = brain_mri.segmentation_table(index, include_negatives=False)
    assert len(seg) == 1800


@pytest.mark.realdata
@needs["kb"]
def test_real_kb():
    kb = optional_sources.load_symptom_kb(CFG.paths["symptom_kb_dir"])
    assert len(kb.patterns) == 304 and kb.patterns["disease"].nunique() == 41
    assert set(optional_sources.HEADACHE_TO_KB.values()) <= set(kb.symptoms)
    assert optional_sources.URGENT_CONDITIONS <= set(kb.patterns["disease"])
