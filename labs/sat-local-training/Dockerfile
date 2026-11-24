FROM python:3.11-slim-bookworm
ARG TORCH_CHANNEL=cpu
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir torch==2.6.0 \
    --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"
WORKDIR /opt/lab
COPY train.py model.py ./
COPY training/ training/
COPY samples/dataset.csv samples/
RUN mkdir /results && chown 10001:10001 /results
USER 10001:10001
CMD ["python", "train.py", "--help"]
