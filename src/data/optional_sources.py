import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

USED_FILES = ("Training.csv", "Symptom-severity.csv", "description.csv")
URGENT_CONDITIONS = frozenset({"Paralysis (brain hemorrhage)", "Heart attack"})
DISEASE_NAME_FIXES = {
    "(vertigo) Paroymsal Positional Vertigo": "(vertigo) Paroxysmal Positional Vertigo",
    "Peptic ulcer diseae": "Peptic ulcer disease",
    "Osteoarthristis": "Osteoarthritis",
    "Dimorphic hemmorhoids(piles)": "Dimorphic hemorrhoids (piles)",
}
SYMPTOM_NAME_FIXES = {"foul_smell_ofurine": "foul_smell_of_urine"}
HEADACHE_TO_KB = {
    "nausea": "nausea",
    "vomitting": "vomiting",
    "sweating": "sweating",
    "visual_symptomps": "visual_disturbances",
    "diplopia": "blurred_and_distorted_vision",
    "speech_disturbance": "slurred_speech",
    "dysarthria": "slurred_speech",
    "motor_weakness": "weakness_in_limbs",
    "hemiplegic": "weakness_of_one_body_side",
    "decreased_consciousness": "altered_sensorium",
    "vertigo": "spinning_movements",
    "ataxia": "loss_of_balance",
    "nasal_congestion": "congestion",
    "rhinorrhoea": "runny_nose",
    "lacrimation": "watering_from_eyes",
    "conjunctival_injection": "redness_of_eyes",
}


def symptom_name(raw: str) -> str:
    s = re.sub(r"_+", "_", re.sub(r"[\s]+", "_", raw.strip().lower())).strip("_")
    return SYMPTOM_NAME_FIXES.get(s, s)


def disease_name(raw: str) -> str:
    s = re.sub(r"\s+", " ", raw.strip())
    return DISEASE_NAME_FIXES.get(s, s)


@dataclass(frozen=True)
class ConditionMatch:
    disease: str
    score: float
    matched: tuple[str, ...]
    description: str
    urgent: bool


@dataclass(frozen=True)
class SymptomKB:
    patterns: pd.DataFrame
    weights: dict[str, int]
    descriptions: dict[str, str]

    @property
    def symptoms(self) -> list[str]:
        return [c for c in self.patterns.columns if c != "disease"]

    def related_conditions(self, symptoms: Iterable[str], top_k: int = 3, min_matched: int = 2) -> list[ConditionMatch]:
        have = {symptom_name(s) for s in symptoms} & set(self.symptoms)
        if len(have) < min_matched:
            return []
        best: dict[str, tuple[float, tuple[str, ...]]] = {}
        for _, row in self.patterns.iterrows():
            pattern = {s for s in self.symptoms if row[s] == 1}
            matched = have & pattern
            if len(matched) < min_matched:
                continue
            w = self.weights
            score = sum(w[s] for s in matched) / sum(w[s] for s in have | pattern)
            if score > best.get(row["disease"], (0.0, ()))[0]:
                best[row["disease"]] = (score, tuple(sorted(matched)))
        ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))[:top_k]
        return [
            ConditionMatch(d, round(s, 3), m, self.descriptions.get(d, ""), d in URGENT_CONDITIONS)
            for d, (s, m) in ranked
        ]


def headache_record_to_kb_symptoms(record: dict) -> set[str]:
    out = {"headache"}
    out |= {kb for field, kb in HEADACHE_TO_KB.items() if str(record.get(field, "")).lower() == "yes"}
    return out


def load_symptom_kb(directory: Path) -> SymptomKB:
    directory = Path(directory)
    missing = [f for f in USED_FILES if not (directory / f).exists()]
    if missing:
        raise FileNotFoundError(f"symptom KB files missing in {directory}: {missing}; see scripts/download_data.md")

    train = pd.read_csv(directory / "Training.csv")
    disease = train.pop("prognosis").map(disease_name)
    sym = train.T.groupby(lambda c: symptom_name(re.sub(r"\.\d+$", "", c))).max().T.astype(int)
    patterns = sym.assign(disease=disease).drop_duplicates().reset_index(drop=True)

    sev = pd.read_csv(directory / "Symptom-severity.csv")
    sev["Symptom"] = sev["Symptom"].map(symptom_name)
    sev = sev[sev["Symptom"] != "prognosis"].drop_duplicates("Symptom")
    weights = dict(zip(sev["Symptom"], sev["weight"].astype(int), strict=True))
    unweighted = set(sym.columns) - set(weights)
    if unweighted:
        raise ValueError(f"symptoms without a severity weight: {sorted(unweighted)}")

    desc = pd.read_csv(directory / "description.csv")
    descriptions = dict(zip(desc["Disease"].map(disease_name), desc["Description"].str.strip(), strict=True))
    return SymptomKB(patterns=patterns, weights=weights, descriptions=descriptions)
