#!/bin/bash

# Clone the repository
git clone https://huggingface.co/spaces/slabstech/dhwani-internal-api-server

cd dhwani-internal-api-server


# Install dependencies
sudo apt-get install -y ffmpeg build-essential

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal

# Set environment variables
export PATH="/root/.cargo/bin:${PATH}"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export HF_HOME=/home/ubuntu/data-dhwani-models
export HF_TOKEN=asdasdadasd

# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --no-cache-dir --upgrade pip setuptools psutil setuptools-rust torch
pip install --no-cache-dir flash-attn --no-build-isolation
pip install --no-cache-dir -r requirements.txt

export HF_HOME=/home/ubuntu/data-dhwani-models
# Run the server
#python src/server/main.py --host 0.0.0.0 --port 7860 --config config_two