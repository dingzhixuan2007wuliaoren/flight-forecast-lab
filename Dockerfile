FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/artifacts/demo

WORKDIR /app
COPY pyproject.toml constraints.txt README.md LICENSE ./
COPY src ./src
COPY artifacts/demo ./artifacts/demo
RUN pip install --no-cache-dir -c constraints.txt . && \
    addgroup --system app && \
    adduser --system --ingroup app app && \
    mkdir -p /app/artifacts/runtime && \
    chown -R app:app /app

USER app

EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn flight_forecaster.api:app --host 0.0.0.0 --port ${PORT:-10000}"]
