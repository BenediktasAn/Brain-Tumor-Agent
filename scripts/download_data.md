# Getting the data

No data is committed to git. Everything lives in `data/raw/` (gitignored), at the paths
listed in `src/config.py`.

## Quick start

```bash
bash scripts/download_data.sh          # about 770 MB in total
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/verify_dataset.py
```

`download_data.sh` uses `kagglehub`. These are public datasets, so no Kaggle account was needed when this was tested.

## Datasets

| Dataset | Kaggle slug / source | Size | Status | Used for |
|---|---|---|---|---|
| Headache Diagnosis | `ashwin0605/headache-diagnosis` | ~1 MB | **required** | headache-type classifier, red-flag rule |
| HeadacheDSS original | github.com/predict-idlab/HeadacheDSS `data/migbase.csv` | 0.1 MB | **required** | marks the 849 real patients in the Kaggle file |
| Brain Tumor: Segmentation & Classification | `indk214/brain-tumor-dataset-segmentation-and-classification` | ~767 MB | **required** | U-Net segmentation, CNN classification |
| Disease Symptom Prediction & Treatment | `sibamsamanta07/disease-symptom-prediction-and-treatment-dataset` | ~2 MB | on (`features.enable_symptom_kb`) | "related conditions" lookup only |
| BraTS 2020 | slug to be supplied in `src/config.py` | ~43 GB | optional, off | glioma deep dive (Phase 7) |
| Brain Tumor Classification (MRI) | `sartajbhuvaji/brain-tumor-classification-mri` | – | optional, off | alternative classifier source |
| LGG Segmentation | `mateuszbuda/lgg-mri-segmentation` | – | optional, off | alternative segmentation source (`two_dataset_mode`) |

## Alternative: Kaggle CLI

Install with `pip install kaggle` and put an API token in `~/.kaggle/kaggle.json`. Then
download each dataset into the folder that `src/config.py` expects, for example:

```bash
kaggle datasets download -d ashwin0605/headache-diagnosis --unzip \
  -p "data/raw/kagglehub/datasets/ashwin0605/headache-diagnosis/versions/1"
kaggle datasets download -d indk214/brain-tumor-dataset-segmentation-and-classification --unzip \
  -p "data/raw/kagglehub/datasets/indk214/brain-tumor-dataset-segmentation-and-classification/versions/1"
kaggle datasets download -d sibamsamanta07/disease-symptom-prediction-and-treatment-dataset --unzip \
  -p "data/raw/kagglehub/datasets/sibamsamanta07/disease-symptom-prediction-and-treatment-dataset/versions/1"
```

If you use other folders, update `PATHS` in `src/config.py` instead.

## Things the Kaggle pages get wrong

These come from inspecting the files; details are in `reports/DATA_ANALYSIS.md`.

**Headache data:**
- The line endings are LF, not CRLF.
- 849 rows are real HeadacheDSS patients and 4,301 rows are generated, including the whole sinus class.
- `previous_attacks` holds random dates.
- The second file, `Oversampled refined Data.csv`, is broken and is not used.

**MRI data:**
- The real layout is `DATASET/Segmentation/<Class>/enh_N(.png|_mask.png)` and `DATASET/classification/{Training,Testing}/<class>/*.jpg`, not `images/` and `masks/`.
- There are 7,023 classification images and 2,192 masks, and no "no tumour" masks.
- Mask background is stored as 3, not 0.
- 507 of the 1,311 test images have near-identical copies in Training.

**Disease-symptom data:**
- Only 304 of the 4,920 rows are distinct.
- `fluid_overload` appears twice, which leaves 131 real symptoms, not 132.
- The treatment and medication files are not used: the assistant never recommends treatment, and some entries are unsafe.
