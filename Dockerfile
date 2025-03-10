FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    sudo \
    wget libvips\
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -ms /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

# Use absolute path for clarity
CMD ["python", "/app/src/server/main.py", "--host", "0.0.0.0", "--port", "7860"]