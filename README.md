# CreditIQ — Calibrated Credit Default Scoring

**CreditIQ** is a machine-learning application that estimates the probability of loan default using applicant and credit-history features. It provides a risk band, a threshold-based decision, and the top contributing risk factors through a Streamlit interface backed by a FastAPI service.

> **Disclaimer:** This is an educational machine-learning project, not a lending tool. The model has not been validated for real-world credit decisions or regulatory compliance.

## Live Demo

- **Streamlit application:** https://creditiq-wzy6ftklskj2tb9zj3kzas.streamlit.app
- **FastAPI backend:** https://creditiq-vbx5.onrender.com
- **API documentation:** https://creditiq-vbx5.onrender.com/docs
- **GitHub repository:** https://github.com/vivekduttsharma2005-hub/creditiq

## Problem Statement

Loan-default prediction is a binary classification problem. CreditIQ estimates the probability that a loan will default and translates the prediction into an understandable risk assessment.

The project focuses on building a complete ML application rather than only training a model: model serving, API integration, a user-facing interface, Docker packaging, and automated testing.

## Key Features

- **Calibrated predictions:** Produces estimated default probabilities using a Platt-calibrated model.
- **Risk classification:** Maps predictions to risk bands and threshold-based outcomes.
- **Risk-factor explanations:** Displays the top contributing risk factors returned by the scoring API.
- **REST API:** Exposes FastAPI endpoints for health checks, individual predictions, and batch scoring.
- **Interactive interface:** Lets users enter applicant information and view the resulting assessment.
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
- **Model:** LightGBM-based model with probability calibration.
- **Calibration:** Platt scaling.
- **Decision threshold:** 0.17 for the deployed model.
- **Feature configuration:** 44 features, without lender-generated columns in the deployed model.

### Data Leakage Prevention

Features are selected with the prediction moment in mind: loan issuance. Variables that reveal post-issuance payment behavior or subsequent loan outcomes are excluded.

Automated leakage tests help prevent excluded information from entering the model features.

### Known Limitations

- The dataset contains accepted loans, so rejected applicants are not represented.
- Recent loans may have unresolved outcomes, creating censoring concerns.
- Calibration and threshold choices do not establish real-world lending suitability.
- Model explanations and predictions should not be treated as causal conclusions.

## Technology Stack

- **Language:** Python
- **Machine learning:** LightGBM, scikit-learn, NumPy, pandas
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

The test suite previously completed with **70 passed**. Test results may change as the code evolves.

## Project Goal

CreditIQ demonstrates an end-to-end ML engineering workflow: leakage-aware feature selection, probability calibration, API development, frontend integration, containerization, automated testing, and public deployment.
