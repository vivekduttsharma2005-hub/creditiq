FROM python:3.11-slim

# LightGBM ko yeh library chahiye (slim image mein nahi hoti)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir --default-timeout=300 -r requirements-api.txt

COPY src/ ./src/
COPY models/calibrated_real.joblib ./models/calibrated_real.joblib

ENV CREDITIQ_MODEL=/app/models/calibrated_real.joblib
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
