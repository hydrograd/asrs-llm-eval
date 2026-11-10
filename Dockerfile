# Reproducible environment for the ASRS LLM evaluation.
#
# Default: run the tests, then rebuild the baselines, results.json and every figure from the
# committed frozen dataset and traces. No API keys or network needed.
#   docker build -t asrs-llm-eval .
#   docker run --rm -v "$PWD/results:/app/results" asrs-llm-eval
#
# Run the next experiment batch (or this week's monitoring panel) with your own keys:
#   docker run --rm -e GROQ_API_KEY -e GEMINI_API_KEY -v "$PWD/results:/app/results" \
#     asrs-llm-eval python scripts/run_experiment.py [--monitor]
#
# Dashboard:
#   docker run --rm -p 8501:8501 asrs-llm-eval streamlit run app/dashboard.py --server.address 0.0.0.0

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Dependencies first, so code edits do not invalidate this layer.
COPY pyproject.toml README.md ./
COPY src/asrs_eval/__init__.py src/asrs_eval/__init__.py
RUN pip install ".[dev,app]"

COPY . .
RUN pip install --no-deps -e . \
    && useradd --create-home --uid 1000 eval \
    && chown -R eval /app
USER eval

CMD ["sh", "-c", "pytest -q && python scripts/run_baselines.py && python scripts/make_report.py"]
