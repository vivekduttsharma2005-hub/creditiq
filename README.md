# CreditIQ — Calibrated Credit Default Scoring

**CreditIQ** is a machine-learning application that estimates the probability of loan default using applicant and credit-history features. It provides a risk band, a threshold-based decision, and the top contributing risk factors through a Streamlit interface backed by a FastAPI service.

> **Disclaimer:** This is an educational machine-learning project, not a lending tool. The model has not been validated for real-world credit decisions or regulatory compliance.

<!-- TODO: add the CI badge once you know the workflow file name:
![CI](https://github.com/vivekduttsharma2005-hub/creditiq/actions/workflows/<workflow-file>.yml/badge.svg) -->

## Live Demo

- **Streamlit application:** https://creditiq-wzy6ftklskj2tb9zj3kzas.streamlit.app
- **FastAPI backend:** https://creditiq-vbx5.onrender.com
- **API documentation:** https://creditiq-vbx5.onrender.com/docs
- **GitHub repository:** https://github.com/vivekduttsharma2005-hub/creditiq

> Both services run on free tiers, so the first request after a period of inactivity can take up to a minute while they wake up.

## Results at a Glance

Evaluated on a held-out test set of **518,706 loans** (22.41% defaulted):

- **Ranking:** ROC-AUC **0.702**, PR-AUC **0.391** (a model with no skill scores 0.224), KS 0.292.
- **Versus a baseline:** LightGBM edges out a logistic regression (ROC-AUC 0.702 vs 0.692); the gain is modest, and the value of the project is the full pipeline around it.
- **Calibration:** Platt scaling cut expected calibration error (ECE) by **42%** on the test set (0.051 to 0.029).
- **Decision:** a cutoff of **0.17** derived from a cost-sensitive analysis reduces expected cost by **42%** compared with approving every loan, under an assumed 5:1 false-negative to false-positive cost ratio. It declines 48.7% of applicants and catches 71.2% of defaulters.
- **Leakage control:** only information available at loan issuance is used; dropping lender-generated columns costs under 0.01 ROC-AUC (0.733 to 0.726 on validation).

The metrics in this README come from the files in [`reports/`](reports/).

## Problem Statement

Loan-default prediction is a binary classification problem. CreditIQ estimates the probability that a loan will default and translates the prediction into an understandable risk assessment.

The project focuses on building a complete ML application rather than only training a model: model serving, API integration, a user-facing interface, Docker packaging, drift monitoring, and automated testing.

## Key Features

- **Calibrated predictions:** Produces estimated default probabilities using a Platt-calibrated model.
- **Risk classification:** Maps predictions to risk bands and threshold-based outcomes.
- **Risk-factor explanations:** Displays the top contributing risk factors returned by the scoring API, written as plain-language reasons.
- **REST API:** Exposes FastAPI endpoints for health checks, individual predictions, and batch scoring.
- **Interactive interface:** Lets users enter applicant information and view the resulting assessment.
- **Drift monitoring:** Population Stability Index (PSI) report across all 44 features.
- **Docker support:** Packages the API and its model artifact into a container.
- **Automated testing:** Includes tests for application behavior and data-leakage controls.

## Architecture

```text
User
  |
  v
Streamlit Frontend
  |
  | HTTP POST /score
  v
FastAPI Backend
  |
  v
Input Validation
  |
  v
Calibrated ML Model
  |
  v
Probability + Risk Assessment
  |
  v
Streamlit Results
```

**Deployment:** Streamlit Community Cloud hosts the frontend, while Render hosts the FastAPI backend.

## Machine Learning Approach

- **Task:** Binary classification of loan default.
- **Target:** `default = 1` for Charged Off and `default = 0` for Fully Paid.
- **Data:** historical loans with resolved Fully Paid / Charged Off outcomes. <!-- TODO: name the dataset, link its source and state its license; add train / validation / test row counts and how the split was made -->
- **Features:** 44 features available at loan issuance, with no lender-generated columns. They include derived features such as `loan_to_income`, `installment_to_income`, `util_x_dti`, `fico_avg` and `credit_history_years`.
- **Model:** LightGBM trained without class weights (877 trees, chosen on validation).
- **Calibration:** Platt scaling.
- **Decision threshold:** 0.17 for the deployed model.

### Model selection

Models were compared on the validation set. The unweighted LightGBM without lender columns was selected. Class weighting lowered PR-AUC (0.375 vs 0.405), and adding lender-generated columns improved ROC-AUC by only 0.0075, so they were left out to keep the model independent of the lender's own risk assessment.

### Calibration

Raw LightGBM scores are not reliable probabilities, so they are calibrated. Platt scaling matched isotonic regression on calibration error (ECE 0.0069 vs 0.0063) without reducing PR-AUC (0.4113 vs 0.4056), and was deployed.

### Decision threshold

The cutoff is cost-sensitive rather than fixed at 0.5. With a false-negative cost of 5 and a false-positive cost of 1, the optimal cutoff on calibrated probabilities is 1 / (1 + 5) = 0.167. The empirically best cutoff on validation data was 0.17. **The 5:1 ratio is an assumption, not real lending economics**; the table below shows how the cutoff moves with it.

### Data Leakage Prevention

Features are selected with the prediction moment in mind: loan issuance. Variables that reveal post-issuance payment behavior or subsequent loan outcomes are excluded.

Automated leakage tests help prevent excluded information from entering the model features.

## Detailed Results

### Final test set (518,706 loans, 22.41% default rate)

| Model | ROC-AUC | PR-AUC | KS | Recall @ 50% precision | Brier | ECE |
|---|---|---|---|---|---|---|
| Base rate (predict the prevalence) | 0.500 | 0.224 | 0.000 | 0.000 | 0.177 | - |
| Logistic regression, no lender columns (class-weighted) | 0.692 | 0.378 | 0.276 | 0.135 | 0.220 | - |
| LightGBM, uncalibrated | 0.702 | 0.391 | 0.292 | 0.150 | 0.161 | 0.051 |
| **LightGBM + Platt scaling (deployed)** | 0.702 | 0.391 | 0.292 | 0.150 | 0.159 | 0.029 |

The class-weighted logistic regression outputs are not calibrated probabilities, so its Brier score is not comparable with the LightGBM rows.

At the 0.17 cutoff on the test set: 48.7% of applicants are declined, 71.2% of defaulters are caught, and expected cost is 337,170 against 581,285 for approving everyone (42.0% lower; cost units, false negative = 5, false positive = 1).

### Model selection (validation set, 20.2% default rate)

| Model | ROC-AUC | PR-AUC | KS | Brier | Trees |
|---|---|---|---|---|---|
| Logistic regression, no lender columns (unweighted) | 0.715 | 0.395 | 0.310 | 0.146 | - |
| LightGBM, no lender columns, class-weighted | 0.700 | 0.375 | 0.283 | 0.157 | 7 |
| **LightGBM, no lender columns, unweighted (selected)** | 0.726 | 0.405 | 0.327 | 0.145 | 877 |
| LightGBM, with lender columns (reference, not deployed) | 0.733 | 0.410 | 0.339 | 0.144 | 391 |

### Calibration (calibration holdout)

| Method | Brier | ECE | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Uncalibrated | 0.1432 | 0.0306 | 0.7297 | 0.4113 |
| **Platt (deployed)** | 0.1415 | 0.0069 | 0.7297 | 0.4113 |
| Isotonic | 0.1415 | 0.0063 | 0.7295 | 0.4056 |

Calibration weakens on the later test set (ECE 0.029 after Platt scaling, versus 0.0069 on the holdout). The default rate also rose from 20.2% on validation to 22.41% on test, which is consistent with a shift in the population and is why drift monitoring matters.

### Threshold analysis (validation)

| FN : FP cost | Theoretical cutoff | Best cutoff | Applicants declined | Defaulters caught | Precision of declines | Cost saving vs approve-all |
|---|---|---|---|---|---|---|
| 2 : 1 | 0.333 | 0.31 | 17.5% | 38.7% | 44.4% | 14.4% |
| 5 : 1 | 0.167 | 0.17 | 47.1% | 73.2% | 31.1% | 40.7% |
| 10 : 1 | 0.091 | 0.09 | 78.1% | 93.7% | 24.0% | 64.0% |

### What drives predictions

| Rank | Feature | Mean abs SHAP | Share of LightGBM gain |
|---|---|---|---|
| 1 | `term_months` | 0.414 | 24.3% |
| 2 | `fico_avg` | 0.239 | 15.4% |
| 3 | `installment_to_income` | 0.213 | 8.5% |
| 4 | `installment` | 0.131 | 3.9% |
| 5 | `loan_amnt` | 0.123 | 2.8% |

Loan term and credit score dominate, and the derived affordability ratio `installment_to_income` ranks third on both measures.

Example outputs, showing the predicted risk with the reasons returned alongside it:

| Predicted default risk | Reasons shown to the user |
|---|---|
| 78.2% | Longer loan term; Loan is large relative to income; High debt-to-income ratio |
| 75.6% | Longer loan term; Loan is large relative to income; High debt-to-income ratio |
| 73.8% | Longer loan term; High debt-to-income ratio; Low credit score |

### Drift monitoring

PSI was computed for all 44 features. On the validation data every feature is stable (maximum PSI 0.056). Under a simulated distribution shift, 5 features (all income- or DTI-related) raise an alert and the other 39 stay stable, so the monitor separates a real shift from noise:

| Feature | PSI under simulated shift | Status |
|---|---|---|
| `dti` | 2.36 | ALERT |
| `annual_inc` | 0.83 | ALERT |
| `loan_to_income` | 0.74 | ALERT |
| `installment_to_income` | 0.72 | ALERT |
| `util_x_dti` | 0.42 | ALERT |

<!-- TODO: say whether drift monitoring runs inside the API or is an offline report, and what the reference data is -->

## Known Limitations

- The dataset contains accepted loans, so rejected applicants are not represented.
- Recent loans may have unresolved outcomes, creating censoring concerns.
- LightGBM improves on a logistic regression baseline only modestly (+0.010 ROC-AUC on the test set).
- Performance and calibration are lower on the test set than on validation, consistent with a shift in the default rate.
- The cutoff declines about half of all applicants at about 31% precision, and the cost saving depends on the assumed 5:1 cost ratio.
- Calibration and threshold choices do not establish real-world lending suitability.
- No fairness or bias audit has been performed, and real credit decisions carry legal requirements (fair-lending rules, adverse-action notices) that this project does not address.
- Model explanations and predictions should not be treated as causal conclusions.

## Technology Stack

- **Language:** Python
- **Machine learning:** LightGBM, scikit-learn, SHAP, NumPy, pandas
- **API:** FastAPI, Uvicorn, Pydantic
- **Frontend:** Streamlit
- **Testing:** pytest, HTTPX
- **Deployment:** Render, Streamlit Community Cloud
- **Containerization:** Docker

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API root |
| GET | `/health` | Health check and model metadata |
| POST | `/score` | Score one applicant |
| POST | `/score/batch` | Score multiple applicants |
| GET | `/docs` | Interactive API documentation |

### Example Request

Send a valid applicant payload to `POST /score`. The following is an abbreviated example; use the API documentation for the complete request schema.

```json
{
  "loan_amnt": 10000,
  "term": 36,
  "installment": 350,
  "annual_inc": 60000,
  "dti": 18.0,
  "revol_bal": 5000,
  "revol_util": 35.0,
  "fico_range_low": 700,
  "fico_range_high": 704
}
```

The API returns a default probability, risk band, threshold-based outcome, and risk-factor information.

## Run Locally

Use Python and a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements-api.txt
python -m pip install -r requirements.txt
```

### Start the API

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to explore the API.

### Start the frontend

In a second terminal, activate the virtual environment and run:

```powershell
python -m streamlit run src/streamlit_app.py
```

The frontend defaults to the deployed Render API. To use a local backend, set `CREDITIQ_API_URL` to `http://127.0.0.1:8000` before starting Streamlit.

## Run with Docker

Build the API image:

```powershell
docker build -t creditiq-api .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 creditiq-api
```

Then open `http://127.0.0.1:8000/health`.

## Tests

Run the automated test suite from the project root:

```powershell
python -m pytest -q
```

The suite contains 70 tests, including data-leakage checks. <!-- TODO: re-run pytest and confirm the count before publishing -->

## Repository Layout

```text
src/        API (api.py) and Streamlit app (streamlit_app.py)
models/     trained, calibrated model artifact
reports/    metrics, calibration, threshold, importance and drift outputs
data/       dataset files
tests/      automated tests, including leakage checks
Dockerfile  container image for the API
```

## Project Goal

CreditIQ demonstrates an end-to-end ML engineering workflow: leakage-aware feature selection, probability calibration, cost-sensitive decision thresholds, drift monitoring, API development, frontend integration, containerization, automated testing, and public deployment.
