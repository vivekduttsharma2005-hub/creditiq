"""Day 8b: the web API.  Run: uvicorn src.api:app --reload
Model file: env CREDITIQ_MODEL (default models/calibrated_real.joblib)"""
import os
from contextlib import asynccontextmanager
from typing import Annotated, List, Literal, Optional
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.service import ScoringService, DISCLAIMER


class Applicant(BaseModel):
    model_config = ConfigDict(extra="forbid")        # unknown fields (e.g. total_pymnt) are REJECTED
    loan_amnt: float = Field(gt=0, le=100_000)
    term_months: Literal[36, 60]
    installment: float = Field(gt=0, le=10_000, description="Monthly payment offered")
    annual_inc: float = Field(ge=0, le=100_000_000)
    dti: Optional[float] = Field(None, ge=-1, le=1000, description="Monthly debt / monthly income, in %")
    revol_util: Optional[float] = Field(None, ge=0, le=1000, description="Credit card utilization, in %")
    revol_bal: float = Field(ge=0)
    delinq_2yrs: int = Field(0, ge=0)
    open_acc: int = Field(ge=0)
    pub_rec: int = Field(0, ge=0)
    total_acc: int = Field(ge=0)
    fico_range_low: int = Field(ge=300, le=850)
    fico_range_high: int = Field(ge=300, le=850)
    emp_years: Optional[float] = Field(None, ge=0, le=10)
    credit_history_years: Optional[float] = Field(None, ge=0, le=80)
    home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER"]
    verification_status: Literal["Verified", "Source Verified", "Not Verified"]
    purpose: str = Field(min_length=2)

    @model_validator(mode="after")
    def fico_order(self):
        if self.fico_range_high < self.fico_range_low:
            raise ValueError("fico_range_high must be >= fico_range_low")
        return self


class ScoreResponse(BaseModel):
    default_probability: float
    risk_band: str
    decision: str
    top_risk_factors: List[str]
    cutoff: float
    model: str
    warning: Optional[str] = None
    disclaimer: str

def create_app(model_path=None, log_path=None):
    path = model_path or os.environ.get("CREDITIQ_MODEL", "models/calibrated_real.joblib")

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = ScoringService(path)     # loaded ONCE at startup, not per request
        yield

    app = FastAPI(title="CreditIQ", version="0.1.0", lifespan=lifespan, description=DISCLAIMER)

    def _score(applicants):
        try:
            return app.state.service.score([a.model_dump() for a in applicants])
        except ValueError as e:                      # Pandera rejection
            raise HTTPException(status_code=422, detail=str(e))

    @app.get("/health")
    def health():
        return {"status": "ok", **app.state.service.info()}

    @app.get("/model-info")
    def model_info():
        return app.state.service.info()

    @app.post("/score", response_model=ScoreResponse)
    def score(applicant: Applicant):
        return _score([applicant])[0]

    @app.post("/score/batch", response_model=List[ScoreResponse])
    def score_batch(applicants: Annotated[List[Applicant], Body(max_length=100)]):
        return _score(applicants)

    return app


app = create_app()
