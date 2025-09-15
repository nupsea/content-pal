FROM python:3.11-slim

WORKDIR /app

# System prep
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pipenv

# Copy Pipfile + lock, install everything into system (may bring CUDA Torch)
COPY Pipfile Pipfile.lock ./
RUN pipenv install --deploy --ignore-pipfile --system

# ✅ Force CPU-only Torch override
RUN python -m pip uninstall -y torch torchvision torchaudio && \
    python -m pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu

# Copy your app code
COPY data/netflix_titles_enriched_full.csv ./data/netflix_titles_enriched_full.csv
COPY src/modules/workflow/ ./src/modules/workflow/
COPY src/modules/search/ ./src/modules/search/

# Sanity check (CPU Torch only)
RUN python -c "import torch; print('torch:', torch.__version__, 'cuda?', torch.cuda.is_available())"

EXPOSE 5001

CMD gunicorn --bind 0.0.0.0:5001 src.modules.workflow.app:app
