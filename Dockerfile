FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY app ./app
COPY eval ./eval
COPY frontend ./frontend
COPY scripts ./scripts
COPY .env.example README.md ./

EXPOSE 8000 8501

# The image ships the lexical + LSA dense path, which needs no model download at runtime.
# For the pretrained bi-encoder, build with requirements-ml.txt and set EMBEDDING_MODEL.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
