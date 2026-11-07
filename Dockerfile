FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    HOST=0.0.0.0 \
    PORT=5000

WORKDIR /app

# 1) Dependencies first: this layer is rebuilt only when requirements-docker.txt changes
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 2) Unprivileged user that owns the app folder
RUN useradd --create-home --uid 10001 appuser && chown appuser:appuser /app

# 3) Source code last: it changes often, so it must not invalidate the layers above
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser templates/ templates/
COPY --chown=appuser:appuser static/ static/

USER appuser
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:5000/health', timeout=5)" || exit 1

CMD ["python", "app.py"]
