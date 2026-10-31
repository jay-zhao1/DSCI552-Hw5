# DSCI 552 Homework 5 - jay-zhao1

All submission content is in English. The original supplied data are used.

## Main deliverables

- `DSCI552_Homework5.ipynb`: complete executed notebook, including analyses and figures.
- `DSCI552_Homework5_Report.pdf`: complete English report derived from the notebook's actual recorded outputs. Correlation plots and other large figures use larger PDF pages to keep their labels readable; zoom or use "fit to page" when printing.
- `Homework 5 Data/`: supplied original raw data, metadata, and decision-tree article.
- `Homework5.pdf`: supplied assignment, retained for reference.
- `requirements.txt`: exact installed Python package versions.
- `scripts/`: notebook builder, clean-kernel runner, PDF renderer and validation tools.
- `results/`: numeric results, CV searches, fold indices, raw-data hashes, rules, coefficients, predictions and validation logs.
- `figures/`: high-resolution PNG and vector PDF figures.

The reproducibility ZIP contains the above, excluding virtual environments, caches and temporary renderings.

## Reproduce

The original execution used Python 3.13 (the exact version appears in `results/versions.csv`).
Run these commands from the extracted project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_notebook.py
python scripts/make_report.py
python scripts/validate_artifacts.py
```

The runner launches a new Jupyter kernel using the same Python interpreter and executes every cell sequentially. A kernel registration is not required. All paths are relative to the project root. Execution time depends on the machine; model searches use one worker and one numerical-computation thread to reduce oversubscription and improve repeatability. On systems where XGBoost requires an OpenMP runtime, install the platform's OpenMP runtime (for example, `brew install libomp` on macOS) if the import reports a missing library. Linux/Windows may need different platform-specific transitive dependencies; `requirements.txt` records the actual macOS environment, not a promise of binary compatibility across platforms.

To regenerate the notebook source itself, optionally run `python scripts/build_notebook.py` before the runner. This replaces the notebook and clears its previous execution outputs. To read or interactively rerun it, use any Jupyter-compatible notebook editor with the same environment. Output figures and result tables are regenerated when it executes.

## Experimental conventions

- Seed 552. Original first 1495 community rows train, last 499 test.
- Training-only five-fold CV selects Ridge/LASSO penalties, PCR component count and XGBoost `reg_alpha`.
- Every fitted imputer, scaler and PCA is inside the model Pipeline and re-fitted within CV folds.
- All six models are frozen before final test evaluation; test outcomes do not revise any model.
- Sample SD uses `ddof=1`; descriptive CVs use imputed training rows and choose exactly 11 predictors.
- Question 1 fits the final trees to all 120 records. Alpha uses stratified five-fold CV and a one-standard-error simplicity criterion.
- Question 2(j) uses the explicitly documented standard `gbtree` interpretation with L1-penalized scalar leaf weights; it does not claim LASSO regressions at tree nodes.
- Main error is MSE; RMSE and R-squared are supplementary.

## AI Usage Disclosure

I used AI tools to help me learn the relevant programming techniques and machine learning concepts, understand code implementations, and assist with debugging.

Native runtime used on this machine: Homebrew `libomp` 23.1.4 on macOS arm64. It is a system dependency, not included in the ZIP or Python requirements. Optional visual QA: `python scripts/render_report.py` renders every PDF page to `tmp/pdfs/report/`; this temporary folder is excluded from the archive.
