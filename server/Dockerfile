# CUDA 12.8 base image for Blackwell (RTX 5070 Ti / sm_120) compatibility.
# Matches the hardware-specific note in docs/INTEGRATION_GUIDE.md §8.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git git-lfs ca-certificates curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Use a venv to isolate from system Python.
RUN python3 -m venv /workspace/venv
ENV PATH="/workspace/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip

# PyTorch with cu128 wheels — required for Blackwell kernels.
RUN pip install --no-cache-dir \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

# LeRobot (with pi0 and dataset extras) + PEFT (LoRA) + WebSocket deps.
# The `dataset` extra pulls in HuggingFace `datasets` which lerobot.datasets imports eagerly.
RUN pip install --no-cache-dir \
    "lerobot[pi,dataset]@git+https://github.com/huggingface/lerobot.git" \
    peft \
    msgpack \
    websockets

COPY packages/policy-client ./packages/policy-client
RUN pip install --no-cache-dir -e /workspace/packages/policy-client

COPY src/lerobot_pi0_policy.py ./src/lerobot_pi0_policy.py
COPY server ./server
COPY runtime_core ./runtime_core

ENV PYTHONPATH="/workspace:/workspace/src:/workspace/packages/policy-client/src"

RUN chmod +x /workspace/server/entrypoint.sh
ENTRYPOINT ["/workspace/server/entrypoint.sh"]
