# ---------- Stage 1: builder ----------
FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim
RUN useradd -m -u 1000 user
COPY --from=builder --chown=user /opt/venv /opt/venv

WORKDIR /home/user/app
ENV PATH="/opt/venv/bin:$PATH" \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence_transformers \
    QDRANT_MODE=memory \
    PYTHONPATH=/home/user/app/src \
    PYTHONUNBUFFERED=1

COPY --chown=user . .
USER user
EXPOSE 7860
CMD ["uvicorn", "felinet.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
