# CUDA 12.2 + cuDNN8 on Ubuntu 22.04 (works with onnxruntime-gpu==1.18.1)
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps (Python 3.10 + build tools + OpenCV/FFmpeg runtime libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential git ca-certificates \
    libgl1 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Your code
COPY handler.py /app/handler.py
COPY README.md  /app/README.md

# Let RunPod runtime know which handler to import: <module>.<function>
ENV RP_HANDLER=handler.handler

# Optional: tweak InsightFace defaults (you can also set these in the Endpoint env)
# ENV IFACE_DET_NAME=buffalo_l
# ENV IFACE_DET_SIZE=640
# ENV IFACE_SWAP_MODEL=inswapper_128.onnx
# ENV JPEG_QUALITY=95

# Start the RunPod serverless worker
CMD ["python3", "-m", "runpod"]
