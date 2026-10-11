FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
# Train first (python scripts/run_pipeline.py) so models/ contains
# model.joblib, metrics.json and metadata.json before building the image.
COPY models/ models/

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--chdir", "app", "app:app"]
