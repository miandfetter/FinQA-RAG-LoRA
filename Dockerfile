# GPU image for Qwen evaluation and LoRA training jobs.
# Built in the cloud with Cloud Build (no Docker needed on your laptop):
#   gcloud builds submit --region=us-central1 --machine-type=e2-highcpu-8 \
#     --tag us-central1-docker.pkg.dev/finqa-rag-lora/finqa/finqa-gpu:v4 .
#
# Based on NVIDIA's CUDA image: it declares NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES,
# which tell the GPU machine to give the container access to its driver. Without them (as in a plain
# python image) PyTorch silently can't see the GPU and runs on the CPU.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/hf \
    TOKENIZERS_PARALLELISM=false

# Ubuntu 24.04's Python is 3.12, matching the project.
# gcc + python3-dev: Triton (used by bitsandbytes, which PEFT imports) compiles a small C helper
# on first use; without a compiler, LoRA setup fails with "Failed to find C compiler".
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-pip gcc python3-dev \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# PyTorch built for CUDA 12.6 (the machine's CUDA 13.0 driver runs it).
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126

COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN pip install -r /tmp/requirements-gpu.txt

# The project code. --no-deps: the laptop-only dependencies (Gemini SDK, FAISS, MLflow) aren't needed here.
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-deps .

ENTRYPOINT ["python", "-m", "finqa_rag_lora.gpu_entry"]