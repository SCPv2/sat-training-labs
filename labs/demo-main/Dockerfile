FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime@sha256:4ec910830ec6c3b837d5621b6f995a31638d3b0824bf0ed4fcaeba7dd4f89b48
WORKDIR /work
COPY requirements-train.lock ./
RUN pip install --break-system-packages --no-cache-dir -r requirements-train.lock
COPY run.sh storage.py settings.py preflight.py train.py ./
ENV PYTHONUNBUFFERED=1
