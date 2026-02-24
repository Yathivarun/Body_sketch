# =============================================================================
# Triton Inference Server — Portrait Sketch Pipeline
# Base: Triton with Python backend support
# PyTorch is installed explicitly BEFORE requirements.txt to prevent pip
# from auto-upgrading/downgrading it when resolving other packages.
# =============================================================================

FROM nvcr.io/nvidia/tritonserver:24.01-py3

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Python backend support for Triton ────────────────────────────────────────
# Triton 24.01 ships Python 3.10; ensure pip is up to date
RUN pip install --no-cache-dir --upgrade pip

# ── Step 1: Pin PyTorch FIRST (CUDA 12.1 wheels matching Triton 24.01 base) ──
# Installing torch separately ensures no other package can silently upgrade it.
RUN pip install --no-cache-dir \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ── Step 2: Install all other project dependencies ────────────────────────────
# torch is already satisfied above; pip will skip it due to version match.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Step 3: ip_adapter — not on PyPI, copied directly into the project ────────
# The ip_adapter/ folder is part of the project source and lands at /app/ip_adapter/
# We add /app to PYTHONPATH so `from ip_adapter.ip_adapter_faceid import ...` resolves.

# ── Step 4: triton python backend client library ──────────────────────────────
RUN pip install --no-cache-dir tritonclient[all]

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Copy project source ───────────────────────────────────────────────────────
# Copies: pipeline/, ip_adapter/, config.py, crops.json
# Model weights and scenes are volume-mounted — NOT baked into the image.
COPY pipeline/         /app/pipeline/
COPY ip_adapter/       /app/ip_adapter/
COPY config.py         /app/config.py
COPY crops.json        /app/crops.json

# Model repository for Triton (backends defined here)
COPY model_repository/ /app/model_repository/

# ── PYTHONPATH: /app so both `pipeline.*` and `ip_adapter.*` resolve ──────────
ENV PYTHONPATH="/app:${PYTHONPATH}"

# ── Expose Triton ports ───────────────────────────────────────────────────────
# 8000 = HTTP, 8001 = gRPC, 8002 = metrics
EXPOSE 8000 8001 8002

# ── Entrypoint ────────────────────────────────────────────────────────────────
# --model-repository: where Triton reads model configs
# --backend-directory: Triton's own backend libs (already in base image)
# --log-verbose=1: helpful during development, remove in production
CMD ["tritonserver", \
     "--model-repository=/app/model_repository", \
     "--log-verbose=1"]
