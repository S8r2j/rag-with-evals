# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /app

# Install dependencies first so code changes don't invalidate this layer.
COPY pyproject.toml README.md ./
RUN mkdir app && touch app/__init__.py \
    && pip install . \
    && rm -rf app

# Bake the tokenizer into the image so containers never download it at runtime.
RUN python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"

COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/uploads \
    && chown -R appuser /app/data
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.asgi:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
