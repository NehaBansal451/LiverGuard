# LiverGuard research app (NOT a clinical system). CPU image. UNTESTED: no Docker daemon was available when this file was written.
# The SMC-LUD dataset is licensed for non-commercial research use: it is never baked into the image; mount it as a volume.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LIVERGUARD_WORKSPACE=/workspace
RUN useradd -m -u 1000 lg && mkdir /workspace && chown lg /workspace
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY --chown=lg configs ./configs
COPY --chown=lg liverguard ./liverguard
COPY --chown=lg scripts ./scripts
USER lg
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"
CMD ["python", "-m", "streamlit", "run", "liverguard/app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
