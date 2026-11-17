# QSAR Modeling Tool

**A Streamlit web app that builds QSAR models for drug discovery: upload a table of molecular descriptors, pick a method, and get a validated model equation with its metrics and plots.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-models-orange)

## What is QSAR?

QSAR (Quantitative Structure–Activity Relationship) modeling predicts how active a molecule will be from numbers that describe its structure, called *descriptors* (molecular weight, LogP, polar surface area, ring counts…). A good QSAR model is a short equation that uses only a few descriptors and still predicts compounds it has never seen. Finding which few descriptors to use, out of dozens or hundreds, is the hard part, and it is what this tool automates.

## Background

I built this during a university research project at Kean University. The research needed a way to develop QSAR models without doing every step by hand, and I was asked to create a tool for it. I studied the modeling and validation methods used in the QSAR literature and designed and implemented the whole application myself: the feature-selection algorithms, the data-splitting methods, the validation metrics and the web interface.

## Features

**Two kinds of model**

| Task | Target | Model | Feature selection |
|---|---|---|---|
| Classification | A category, e.g. Active / Inactive | Linear Discriminant Analysis (LDA) | Genetic Algorithm (GA-LDA) or Best Subset Selection (BSS-LDA) |
| Regression | A number, e.g. pIC50 | Multiple Linear Regression (MLR) | Genetic Algorithm (GA-MLR) or Best Subset Selection (BSS-MLR) |

- **Genetic algorithm feature selection,** implemented from scratch: population initialization, crossover, mutation and fitness evaluation, with population size, generations, mutation and crossover probability as settings.
- **Best subset selection,** which tries every combination of descriptors of the chosen equation length.
- **Three ways to split training and test data:** random, Kennard-Stone (picks a training set that covers the descriptor space) and a cluster-based split.
- **Data pre-treatment:** drops descriptors with near-zero variance and one of each pair of highly inter-correlated descriptors (configurable cut-offs).
- **QSAR validation metrics:**
  - Regression: R², Q² leave-one-out, r²m (average), Q²F1, Q²F2, MAE for training and test, and a MAE-based prediction quality score.
  - Classification: accuracy, precision, recall (sensitivity), specificity, F1 and MCC for training and test, plus the confusion matrix.
- **Plots:** Williams plot (applicability domain) and observed-vs-predicted for regression; confusion matrix and ROC curve for classification.
- **Readable output:** the final model is shown as an equation with its selected descriptors and coefficients.

## Try it with the sample data

The `examples/` folder has two small **synthetic** datasets (80 made-up compounds, 11 descriptors), so you can try the app without real research data:

| File | Use with | Target column |
|---|---|---|
| [`sample_regression.csv`](examples/sample_regression.csv) | Regression (GA-MLR, BSS-MLR) | `pIC50` |
| [`sample_classification.csv`](examples/sample_classification.csv) | Classification (GA-LDA, BSS-LDA) | `Activity` |

The activity was generated from three descriptors (`LogP`, `TPSA`, `AromaticRings`) plus noise, so a good run should find those three. The data also contains one near-duplicate descriptor (`MolMR`) and one nearly constant descriptor (`Halogens`) for the pre-treatment step to remove. The script that made them is [`examples/generate_sample_data.py`](examples/generate_sample_data.py).

## Run locally

Requirements: Python 3.10+.

```bash
git clone https://github.com/Brayanketchup/qsar-modeling-tool.git
cd qsar-modeling-tool
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt openpyxl
streamlit run app.py
```

The app opens at http://localhost:8501. (`openpyxl` is only needed to upload `.xlsx` files.)

**Steps in the app:**

1. Upload a CSV or XLSX file.
2. Choose **Regression** or **Classification**, then the split method and the model (GA or BSS).
3. Optionally enable data pre-treatment and set its cut-offs.
4. Set the equation length and test-set size in the sidebar (and the GA settings, if you chose GA).
5. Press **Run** and read the equation, metrics and plots.

## Dataset format

| Column | Content |
|---|---|
| 1st | Compound ID or name (not used by the model) |
| 2nd | Target: a number for regression, a class label for classification |
| 3rd onward | Descriptors, one numeric column each (at least 4) |

## Project structure

```text
app.py                 Entry point (streamlit run app.py)
gui.py                 Streamlit interface: upload, settings, which method to run
GA_LDA.py / GA_MLR.py  Genetic algorithm feature selection for LDA / MLR
BSS_LDA.py / BBS_MLR.py  Best subset selection for LDA / MLR
fitness_function.py    Leave-one-out validation, r²m, MAE-based fitness and prediction quality
division_methods.py    Kennard-Stone and cluster-based train/test splits
preTreatment.py        Variance and inter-correlation descriptor filters
examples/              Synthetic sample datasets and the script that generates them
```

## Versions

- **v1.0, research version (this release).** The code as it was written during the research project, kept unchanged.
- **v2, planned.** A cleanup release: remove leftover commented-out code and debug output, clearer labels in the interface, tighter input limits, unit tests for the metric and splitting functions, and a hosted live demo.

**Known limitations in v1.0** (to be fixed in v2):

- When data pre-treatment removes descriptors, the names shown in the final equation are not re-aligned with the remaining columns. Run without pre-treatment to get correct descriptor names.
- The test-size slider goes up to 1.0; keep it between 0.1 and 0.4.
- The "Euclidean" split is a cluster-stratified random split (KMeans), not a distance-based selection.

## Tech stack

Python, Streamlit, scikit-learn, pandas, NumPy, SciPy, Matplotlib, Seaborn.
