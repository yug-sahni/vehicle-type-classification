# Vehicle Type Classification — ML Course Project I (Classification)



Predict `vehicle_class` (0 / 1 / 2) from 20 numeric vehicle specifications.
Models used (as taught in the course): Logistic Regression, Decision Tree, Random Forest, Naive Bayes.

## Results

| Model | 5-fold CV accuracy | Kaggle public score | Submission file |
|---|---|---|---|
| **Logistic Regression** | 0.8943 | **0.903** | `submissions/submission_logistic_regression.csv` |
| Random Forest | 0.8848 | 0.892 | `submissions/submission_random_forest.csv` |
| Naive Bayes (tuned, PCA-whitened) | 0.8628 | 0.868 | `submissions/submission_naive_bayes.csv` |
| Decision Tree | 0.8641 | 0.866 | `submissions/submission_decision_tree.csv` |
| Naive Bayes (plain Gaussian) | 0.8475 | 0.847 | `submissions/submission_naive_bayes_plain.csv` |

**Best model (public leaderboard): Logistic Regression** (L2, C = 0.3, standardised features) — public score 0.903.

## Repository layout

```
.
├── 1_logistic_regression.ipynb   # tunes + fits Logistic Regression, writes its submission
├── 2_decision_tree.ipynb         # same for Decision Tree
├── 3_random_forest.ipynb         # same for Random Forest
├── 4_naive_bayes.ipynb           # same for Naive Bayes (+ plain variant)
├── plot_data.ipynb               # EDA / report figures -> figures_report/
├── data/                         # competition files (train, test, sample submission)
├── submissions/                  # one CSV per model (+ small *_result.json with CV scores/params)
├── figures_report/               # PNG figures used in the report
├── report/                       # final report (.docx / .pdf), results table, Kaggle screenshot
└── requirements.txt
```

## How to reproduce

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 1_logistic_regression.ipynb   # ~1 min
jupyter nbconvert --to notebook --execute --inplace 2_decision_tree.ipynb         # ~1 min
jupyter nbconvert --to notebook --execute --inplace 4_naive_bayes.ipynb           # ~2 min
jupyter nbconvert --to notebook --execute --inplace 3_random_forest.ipynb         # ~8 min
jupyter nbconvert --to notebook --execute --inplace plot_data.ipynb               # ~1 min
```

Run from the repository root (the notebooks look for the CSVs in `./data`). All randomness uses seed 42.

## Approach in brief

* **Preprocessing:** drop `vehicle_id`; drop the 2,500 exact duplicate training rows (otherwise CV leaks across folds and is inflated by ~4.7 points); standardise inside the CV pipeline where needed (Logistic Regression, Naive Bayes). No imputation, outlier removal or resampling was needed.
* **Tuning:** cross-validated grid / randomised search (5-fold stratified, accuracy), then a repeated-CV stability check and a train-vs-CV overfitting check for each model.
* **Finding:** all models plateau at about 0.86–0.90. Linear boundaries do best and heavier models add nothing, so Logistic Regression is the final model. Full details are in `report/Report_Vehicle_Classification_Final.pdf`.
