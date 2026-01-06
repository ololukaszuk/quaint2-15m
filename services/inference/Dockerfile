FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

COPY ../shared /app/shared
COPY requirements.txt .
COPY *.py ./

RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

RUN mkdir -p /app/logs /app/models/active

CMD ["python", "-u", "app.py"]
