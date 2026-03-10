# ── Imagen base ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Variables de entorno ───────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_SYSTEM_PYTHON=1

# ── Instalar uv ────────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ── Directorio de trabajo ──────────────────────────────────────────────────────
WORKDIR /app

# ── Dependencias: instalar antes de copiar el código (mejor cache) ─────────────
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

# ── Código de la aplicación ────────────────────────────────────────────────────
COPY src/app.py ./app.py
COPY models/ ./models/

# ── Carpetas que la app necesita en runtime ────────────────────────────────────
RUN mkdir -p data .streamlit

# ── Puerto Streamlit ───────────────────────────────────────────────────────────
EXPOSE 8501

# ── Healthcheck ────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Arranque ───────────────────────────────────────────────────────────────────
CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]