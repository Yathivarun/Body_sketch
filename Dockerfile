# =============================================================================
# Triton Inference Server — Portrait Sketch Pipeline
# Base: Triton 24.01 / Python 3.10 / CUDA 12.1
#
# Install order matters:
#   1. System libs
#   2. torch + torchvision (pinned, must come before everything else)
#   3. requirements.txt (all deps except rembg and tritonclient)
#   4. rembg --no-deps (prevents it pulling a conflicting onnxruntime)
#   5. tritonclient
#   6. Copy source code
# =============================================================================

FROM nvcr.io/nvidia/tritonserver:24.01-py3

# ── Step 1: System dependencies ───────────────────────────────────────────────
# libgl1 + libglib2.0 required by opencv-python-headless at runtime
# libgomp1 required by insightface / onnxruntime
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

RUN pip install --no-cache-dir --upgrade pip

# ── Step 2: PyTorch — pinned BEFORE requirements.txt ─────────────────────────
# Must come first so nothing in requirements.txt can override the version.
# cu121 wheels match the CUDA 12.1 runtime in Triton 24.01 base image.
RUN pip install --no-cache-dir \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ── Step 3: All project dependencies ─────────────────────────────────────────
# rembg intentionally excluded here — installed separately in Step 4.
# torch/torchvision already satisfied above; pip will not touch them.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Step 4: rembg — installed with --no-deps ──────────────────────────────────
# rembg's setup.py pulls onnxruntime (CPU) as a hard dependency which would
# conflict with onnxruntime-gpu already installed above.
# --no-deps skips that pull entirely. All of rembg's actual runtime needs
# (numpy, pillow, opencv, onnxruntime-gpu, pooch, PyMatting) are already
# present from Step 3.
RUN pip install --no-cache-dir --no-deps rembg==2.0.69

# ── Step 5: Triton client library ─────────────────────────────────────────────
RUN pip install --no-cache-dir tritonclient[all]

# ── Step 6: Copy project source ───────────────────────────────────────────────
# Only source code is copied into the image.
# Model weights, scenes, crops.json are volume-mounted at runtime.
WORKDIR /app

COPY pipeline/         /app/pipeline/
COPY ip_adapter/       /app/ip_adapter/
COPY model_repository/ /app/model_repository/
COPY config.py         /app/config.py
COPY crops.json        /app/crops.json

# ── PYTHONPATH ────────────────────────────────────────────────────────────────
# /app resolves both `pipeline.*` and `ip_adapter.*` imports
ENV PYTHONPATH="/app:${PYTHONPATH}"

# ── Prevent runtime downloads ─────────────────────────────────────────────────
# All models must be volume-mounted. These flags make missing models
# fail loudly at startup rather than silently downloading.
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# ── Triton ports ──────────────────────────────────────────────────────────────
EXPOSE 8000 8001 8002

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["tritonserver", \
     "--model-repository=/app/model_repository", \
     "--log-verbose=1"]
