FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8000

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY web ./web
COPY scripts ./scripts

# Train at build time so the container starts instantly with a ready model.
RUN python scripts/train.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn solarsense.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
