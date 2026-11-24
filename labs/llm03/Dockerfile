FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime@sha256:4ec910830ec6c3b837d5621b6f995a31638d3b0824bf0ed4fcaeba7dd4f89b48
WORKDIR /lesson
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY checkpointing.py storage.py settings.py train_llm03.py train.py ./
ENV PYTHONUNBUFFERED=1
