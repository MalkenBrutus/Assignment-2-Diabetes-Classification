# Assignment 2: Diabetes Classification

IE500618 Machine Learning, NTNU. We explore the supplied survey features, fit Logistic Regression, and interpret errors at thresholds 0.50 and 0.40.

Group members: Jørund Sæten Birkeland, Matteo Castelli, Haakon Torvik Gullbekk, and Marius Alexander Langaas Klæbo.

## Files

- `Assignment_2_Diabetes_Classification.ipynb`: executed analysis, all 21 predictors, visualisations, and model evaluation.
- `Diabetes_Classification_Written_Answers.docx` and `.pdf`: written answers to Q1-Q7 and G1-G2.
- `Diabetes_Classification_Written_Answers.md`: readable text version of the same answers.
- `diabetes_binary_classification_data.csv`: supplied assignment dataset, unchanged.
- `results/`: class counts, split counts, feature guide, EDA summaries, and evaluation tables.
- `figures/`: the figures displayed in the notebook.
- `requirements.txt`: analysis versions and notebook tools.

## Run the notebook

Use Python 3.12.14 (Python 3.12). From this repository folder:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install and start Jupyter:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab Assignment_2_Diabetes_Classification.ipynb
```

Select the environment's Python kernel. Restart it and run all cells in order. Keep the CSV beside the notebook. The notebook recreates the tables in `results/` and figures in `figures/`; it does not regenerate the written document. If you change the analysis, revise the written answers to match its new results.

## Dataset and method

The file contains 253,680 rows, 21 predictors, and `Diabetes_binary`. Under the supplied course guide, class 0 means no diabetes and class 1 means prediabetes or diabetes. There are no missing cells. Identical recorded rows are retained as instructed.

The notebook uses one 80/20 stratified split with random_state=42. Detailed EDA uses TRAIN only. The pipeline standardises numerical variables, explicitly encodes ordered categories, and passes binary features through as 0/1. Logistic Regression is fitted once on TRAIN with max_iter=2000. TEST is not resampled or used for model selection. Both prescribed thresholds use the same probabilities.

## TEST results

| Threshold | Accuracy | Precision | Recall | F1 | False positives | False negatives |
|---|---|---|---|---|---|---|
| 0.50 | 0.8621 | 0.5163 | 0.1587 | 0.2428 | 1,051 | 5,947 |
| 0.40 | 0.8564 | 0.4731 | 0.2699 | 0.3437 | 2,125 | 5,161 |

The always-class-0 reference has 86.07% TEST accuracy and zero recall. Lowering the threshold finds additional positive cases and produces more false positives. Neither threshold is clinically validated.

## Sources and interpretation

The assignment and supplied *Diabetes Classification: Understanding the Dataset* guide govern the target meanings and feature codes. Preprocessing and threshold comparisons follow the course's *Heart Disease Classification - Lecture Code* and classification slides. The course archive and slides are not redistributed here. Our [Assignment 1](https://github.com/MalkenBrutus/housing-price-prediction-group-26) provides presentation continuity; no lecturer feedback has been assumed.

The contribution reflection follows the supplied group scenario. It is not a verified record of meetings or individual commits.

These are survey-based associations and performance on one file split. The target merges prediabetes and diabetes. Ordered coding imposes equal category steps. This analysis is not a clinical diagnostic tool, causal study, or survey-weighted population estimate.

CSV SHA-256: `19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0`
