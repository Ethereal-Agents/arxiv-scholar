FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Pre-compile bytecode to reduce cold start times on Cloud Run
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy pyproject.toml and uv.lock first to leverage Docker build layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync without the project itself
# --frozen ensures we install exactly what's in the lockfile
# --no-dev excludes testing and development tools
# --no-install-project prevents it from looking for src/
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the application code
COPY . /app

# Install the project itself now that src/ is present
RUN uv sync --frozen --no-dev

# Add the uv managed virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Cloud Run / Hugging Face inject the PORT environment variable dynamically.
# Hugging Face defaults to 7860. We use a shell fallback to 7860 if not set.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn arxiv_scholar.api.server:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips='*'"]
