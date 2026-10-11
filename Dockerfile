FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY settings.py config.yaml ./
COPY crypto crypto
COPY server server
COPY client client
COPY preprocessing preprocessing
COPY detectors detectors

# Model weights and server state are mounted, not baked in (see docker-compose.yml).
ENV PQBIO_CONFIG=/app/config.yaml
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
