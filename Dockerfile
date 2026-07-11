FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/artifacts/demo

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . && \
    python -m flight_forecaster train-demo --output "$MODEL_DIR" --price-rows 5000 --ontime-rows 7000

EXPOSE 8000
CMD ["uvicorn", "flight_forecaster.api:app", "--host", "0.0.0.0", "--port", "8000"]
