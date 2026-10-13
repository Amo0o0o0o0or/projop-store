# Wisconsin Diagnostic Explorer

An end-to-end, local-first data science portfolio project. It explores the Breast Cancer Wisconsin Diagnostic dataset, compares two classifiers with training-only cross-validation, evaluates the selected model on a held-out test set, and presents the results in an interactive Flask dashboard.

## Why this project

- Runs locally and costs nothing; the dataset ships with scikit-learn, so no API key, cloud account, or separate data download is needed.
- Demonstrates an end-to-end workflow: data inspection, stratified splitting, preprocessing, model comparison, evaluation, threshold analysis, and model interpretation.
- Keeps the test set out of model selection to reduce evaluation leakage.
- Treats malignant cases as the positive class and reports recall, specificity, precision, ROC-AUC, average precision, and a confusion matrix.

## Run locally

Requires Python 3.10 or later.

```bat
"%APPDATA%\uv\python\cpython-3.14-windows-x86_64-none\python.exe" -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python webapp.py
```

If that Python path does not exist on another machine, install Python from [python.org](https://www.python.org/downloads/) and select **Add python.exe to PATH** during setup. Then `python -m venv .venv` can be used for the first command.

In PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead. After activating `.venv`, run `python webapp.py`.

The dashboard opens at `http://127.0.0.1:5000`. Plotly charts load their JavaScript library from its CDN, so the browser needs internet access to display charts. To run the checks:

```powershell
pytest -q
```

## Project structure

```text
webapp.py           Flask routes and visualizations
templates/index.html Flask dashboard interface
model.py            Dataset loading, model selection, evaluation
tests/test_model.py Focused checks for labels, metrics, and training
```

## Method

1. Load the 569-row, 30-feature dataset packaged with scikit-learn. The positive label is malignant.
2. Create one stratified 80/20 train/test split with a fixed random seed.
3. Compare scaled logistic regression and random forest using five-fold stratified cross-validation on the training data only.
4. Refit the highest mean CV ROC-AUC model on all training rows and evaluate it once on the held-out test set.
5. Use permutation importance on the test split as a model-specific exploratory explanation. The dashboard also shows how a decision threshold changes classification outcomes.

## Interview walkthrough

> I built a local-first diagnostic classification case study using scikit-learn's packaged Wisconsin dataset. I treated malignant as the positive class because missing a positive case has a different cost from raising a false alarm, so I report recall and specificity alongside ROC-AUC and precision. I compare logistic regression and random forest with stratified cross-validation on the training split, then keep a separate test split for the final evaluation. The dashboard makes the threshold trade-off and the model's permutation importance visible. This is an educational exercise, not a clinical tool.

Be ready to discuss why accuracy alone is insufficient, how stratification helps with class balance, why preprocessing belongs inside a pipeline, what a false negative means here, and why these results do not establish clinical validity.

## Data source and limitations

The dataset is the Breast Cancer Wisconsin (Diagnostic) dataset, originally collected by W. Nick Street, William H. Wolberg, and Olvi L. Mangasarian and made available through the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). This project uses the copy distributed by scikit-learn.

The dataset is small, old, and not representative of all patients or clinical settings. Its labels and feature-generation process are inherited from the source. Metrics from this exercise must not be interpreted as evidence of safety, fairness, or real-world diagnostic performance. This model is not for clinical use.
