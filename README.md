# Wine Cultivar Discriminant Analysis

A reproducible statistical analysis of whether **13 chemical measurements can be compressed into a two-dimensional supervised representation while preserving separation among three wine cultivars**.

This repository is framed as an independent multivariate-analysis project rather than an academic exercise. Linear Discriminant Analysis (LDA) is used both as a supervised dimensionality-reduction method and as a classifier.

## Research question

> How much can the chemical profile of a wine be compressed while retaining the information that distinguishes its cultivar?

The dataset contains **178 observations**, **13 numeric predictors**, **3 cultivar classes**, **0 missing values**, and **0 duplicate rows**. Because LDA can produce at most `min(p, K-1)` discriminant axes, this problem reduces naturally from **13 original dimensions to 2 linear discriminants**.

## Main findings

- Original feature space: **13 dimensions**.
- LDA representation: **2 dimensions**.
- LD1: **68.75%** of discriminant variance.
- LD2: **31.25%**.
- The two axes therefore span **100% of the available LDA discriminant space**.
- Stratified 10-fold cross-validation accuracy: **99.41% ± 1.86%**.
- Stratified holdout accuracy: **100.00%** on 45 test observations.
- The strongest contributors to the discriminant space include **flavanoids, proline, color intensity, OD280/OD315, and alcohol**.

The perfect holdout result is treated as a sample-specific outcome, not as proof of perfect generalization; cross-validation is used as the more stable performance summary.

## Project structure

```text
wine_lda_portfolio/
├── data/
│   ├── raw/Wine.csv
│   └── processed/wine_clean.csv
├── docs/
│   ├── methodology.md
│   └── variable_dictionary.md
├── notebooks/
│   ├── 01_data_audit_and_eda.ipynb
│   ├── 02_lda_dimensionality_reduction.ipynb
│   └── 03_validation_and_interpretation.ipynb
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   ├── config.py
│   ├── data.py
│   ├── modeling.py
│   ├── visualization.py
│   └── run_analysis.py
├── tests/test_pipeline.py
├── run_analysis.py
├── requirements.txt
└── pyproject.toml
```

## Reproduce the analysis

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the complete pipeline

```bash
python run_analysis.py
```

### 4. Run tests

```bash
pytest -q
```

### 5. Explore the notebooks

Open `notebooks/` in VS Code or Jupyter. The notebooks are intentionally separated into data audit, LDA interpretation, and predictive validation.

## Analysis design

The raw Kaggle CSV is preserved unchanged. Since the file contains no header, the ingestion layer assigns the canonical Wine Recognition names. The features are standardized before LDA so that coefficient magnitudes are easier to compare across measurements with different units.

The descriptive LDA projection is fitted on the full clean dataset to study class geometry. Predictive performance is evaluated separately with a stratified holdout set and stratified cross-validation, avoiding the common mistake of reporting in-sample separation as predictive accuracy.

## Key outputs

- `outputs/figures/lda_projection.png` — two-dimensional LDA map of the three cultivars.
- `outputs/figures/feature_discriminant_strength.png` — variables contributing most strongly to the LDA space.
- `outputs/figures/confusion_matrix.png` — holdout classification results.
- `outputs/tables/lda_scalings.csv` — coefficients defining LD1 and LD2.
- `outputs/tables/model_metrics.csv` — dimensionality and validation metrics.
- `outputs/tables/run_summary.json` — compact machine-readable summary.

## Scope

The analysis demonstrates supervised dimensionality reduction and cultivar discrimination within this dataset. It does not establish causal chemical mechanisms, nor does it imply that the model will transfer unchanged to wines from different regions, vintages, laboratory protocols, or cultivars.

## Tools

Python · pandas · NumPy · scikit-learn · Matplotlib · pytest · Jupyter
