#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export KAGGLEHUB_CACHE="$PWD/data/raw/kagglehub"
mkdir -p "$KAGGLEHUB_CACHE" data/raw/headachedss

DATASETS=(
  "ashwin0605/headache-diagnosis"
  "indk214/brain-tumor-dataset-segmentation-and-classification"
  "sibamsamanta07/disease-symptom-prediction-and-treatment-dataset"
)

for slug in "${DATASETS[@]}"; do
  echo ">> $slug"
  "$PY" -I -c "import kagglehub, sys; print(kagglehub.dataset_download(sys.argv[1]))" "$slug"
done

echo ">> HeadacheDSS migbase.csv"
curl -sSfL -o data/raw/headachedss/migbase.csv \
  https://raw.githubusercontent.com/predict-idlab/HeadacheDSS/master/data/migbase.csv

echo "done. Next: $PY scripts/prepare_data.py && $PY scripts/verify_dataset.py"
