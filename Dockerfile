FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Pre-compile bytecode to reduce cold start times on Cloud Run
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy pyproject.toml and uv.lock first to leverage Docker build layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync
# --frozen ensures we install exactly what's in the lockfile
# --no-dev excludes testing and development tools
RUN uv sync --frozen --no-dev

# Copy the rest of the application code
COPY . /app

# Add the uv managed virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Cloud Run injects the PORT environment variable dynamically.
# Using sh -c allows us to evaluate ${PORT} and fallback to 8080 if not set.
CMD ["sh", "-c", "uvicorn arxiv_scholar.api.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
