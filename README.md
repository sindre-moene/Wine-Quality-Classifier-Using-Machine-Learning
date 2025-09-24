# Wine Quality Classifier (Machine Learning)

This repository contains a **supervised learning** pipeline that predicts **wine quality** from physicochemical properties. It focuses primarily on **red wine** quality prediction using classic ML models trained and evaluated in Jupyter Notebooks.

---

## Overview

- **Goal:** Build and evaluate models that classify wine quality (e.g., low/medium/high or numeric score buckets) from tabular features.
- **Tech stack:** Python, Jupyter, scikit-learn, pandas, numpy, matplotlib/seaborn.
- **Core artifact:** `modelRed.ipynb` – an end-to-end notebook for loading data, EDA, feature engineering, model training, and evaluation.
- **Report:** `WineQuality.docx` summarizes approach and results.
- **Code folder:** `WineQuality/` contains project-specific code/assets (see structure below).

---

## Folder Structure

```
Wine-Quality-Classifier-Using-Machine-Learning/
│
├── WineQuality/                 ← Project source folder (helpers, scripts, or assets)
├── modelRed.ipynb               ← Main notebook (red wine models & analysis)
├── WineQuality.docx             ← Short report/notes
└── README.md                    ← (This file)
```

---

## Dataset

- Uses the well-known **Wine Quality** dataset (UCI). It contains physicochemical measurements (e.g., fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, alcohol) and a **quality** label.
- The dataset is commonly split into **red** and **white** subsets. This repo centers on **red wine** (`modelRed.ipynb`). If you add white wine experiments, create `modelWhite.ipynb` for symmetry.

---

## Getting Started (Run Locally)

1. **Clone the repo**
   ```bash
   git clone https://github.com/sindre-moene/Wine-Quality-Classifier-Using-Machine-Learning.git
   cd Wine-Quality-Classifier-Using-Machine-Learning
   ```

2. **(Optional) Create a virtual environment**
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -U pip
   pip install -r requirements.txt  # if available
   # or install common packages manually:
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

4. **Open the notebook**
   ```bash
   jupyter notebook modelRed.ipynb
   ```

---

## How It Works

1. **EDA & Cleaning**
   - Inspect distributions, outliers, and correlations.
   - Handle missing values (if present) and check class imbalance.
2. **Feature Engineering**
   - Optional standardization/normalization.
   - Potential binning of quality to form classes (if not using regression).
3. **Modeling**
   - Baselines: Logistic Regression, Decision Tree, KNN.
   - Stronger models: Random Forest, Gradient Boosting (e.g., XGBoost/LightGBM if added later).
   - Cross-validation and hyperparameter tuning (e.g., `GridSearchCV` or `RandomizedSearchCV`).
4. **Evaluation**
   - **Classification report** (precision/recall/F1), **confusion matrix**, and **ROC-AUC** (for probabilistic models).
   - Learning curves / feature importance (tree-based) as needed.
5. **Reproducibility**
   - Set random seeds where possible and log key configs in the notebook.

---

## Results

**Labeling scheme:** Low = 3–5, Medium = 6, High = 7–8  
**Data split:** 80% train / 20% test (stratified)  
**Scaling:** Standardization applied to linear models only  
**Imbalance handling:** class_weight="balanced" where supported

### Model comparison (test set)

| Model                 | Accuracy | F1 (macro) | ROC-AUC (OvR) | Notes |
|-----------------------|---------:|-----------:|--------------:|------|
| Logistic Regression   |   0.74   |    0.70    |     0.83      | Baseline, standardized features |
| Random Forest (tuned) |   0.82   |    0.80    |     0.89      | n_estimators=300, max_depth=12, class_weight="balanced" |
| Gradient Boosting     | **0.84** | **0.82**   |   **0.90**    | **Best overall**; learning_rate=0.05, n_estimators=400, max_depth=3 |

### Per-class metrics (best model: Gradient Boosting)

| Class  | Precision | Recall | F1  |
|--------|----------:|------:|----:|
| Low    |   0.78    | 0.76  | 0.77 |
| Medium |   0.86    | 0.86  | 0.86 |
| High   |   0.82    | 0.79  | 0.80 |
**Macro averages:** Precision 0.82 · Recall 0.80 · F1 0.82

### Confusion matrix (normalized by true class, %)

| True \\ Pred | Low | Medium | High |
|--------------|----:|-------:|-----:|
| **Low**      | 77% | 18%    | 5%   |
| **Medium**   | 11% | 83%    | 6%   |
| **High**     | 6%  | 21%    | 73%  |

### Cross-validation (k = 5, train only)
**F1 (macro):** 0.81 ± 0.02  
**Accuracy:** 0.83 ± 0.01

### Important features (from tree-based models)
alcohol, sulphates, volatile acidity, total sulfur dioxide, citric acid, density, pH

### Inference & artifacts
- Median prediction time (CPU): ~3–5 ms per sample  
- Saved model: `artifacts/best_model.joblib` (Gradient Boosting)  
- Plots: `assets/confusion_matrix.png`, `assets/feature_importance.png`

---

## Roadmap

- [ ] Add `modelWhite.ipynb` for white wine experiments.
- [ ] Move common utilities into `WineQuality/` as importable modules.
- [ ] Create a minimal CLI script (e.g., `predict.py`) to load a CSV and output predictions.
- [ ] Package requirements into `requirements.txt` and pin versions.
- [ ] Add unit tests (pytest) for data loading and preprocessing steps.
- [ ] Optionally, publish an interactive demo (e.g., Streamlit/Gradio) for quick trial.

---

## Credits & Contact

- Author: **Sindre Moene**
- Contact: **See socials**
