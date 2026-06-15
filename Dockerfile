# Reproducible CPU-only environment for the offline reproduction path (reproduce.sh).
# No GPU, no API keys: builds the reasoner + tests + cached-transcript aggregation.
#
#   docker build -t abduction-dissociation .
#   docker run --rm abduction-dissociation            # runs reproduce.sh (CPU-only, ~15-20 min)
#   docker run --rm abduction-dissociation pytest -q  # just the tests
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first (smaller, no CUDA), so the package install finds it satisfied.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir -e .

ENV OMP_NUM_THREADS=2
CMD ["bash", "reproduce.sh"]
