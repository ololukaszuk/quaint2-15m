FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .

# Install PyTorch nightly with CUDA 12.8 support
RUN pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

RUN mkdir -p /app/logs /app/models/active /app/models/archive

CMD ["python", "-u", "train.py"]