#!/bin/bash
set -euo pipefail

echo "=== Mamba-3 + Flash-Attn setup ==="

# Flash-attn 2.8.3 (cp312 wheel — requires Python 3.12)
echo "[1/3] Installing flash-attn..."
pip install --break-system-packages --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Build dependencies for mamba-ssm
echo "[2/3] Installing mamba-ssm build dependencies..."
pip install --break-system-packages --no-cache-dir einops ninja packaging transformers

# mamba-ssm from source (required for Mamba3 support)
echo "[3/3] Building mamba-ssm from source..."
MAMBA_FORCE_BUILD=TRUE pip install --break-system-packages --no-cache-dir git+https://github.com/state-spaces/mamba.git --no-build-isolation

# Verify
python3 -c "
from mamba_ssm.modules.mamba3 import Mamba3
from flash_attn.losses.cross_entropy import CrossEntropyLoss
print('mamba-ssm + flash-attn OK')
"

echo "=== Setup complete ==="
