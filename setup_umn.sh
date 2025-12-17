#!/usr/bin/env bash
# setup_umn.sh
# Reproduces the UMN/MSI micromamba environment used for MINT experiments,
# pinned to the exact package versions listed below.
#
# Prereq (run these in your session BEFORE calling this script):
#   module load gcc/8.2.0
#   module load ompi/3.1.6/gnu-8.2.0
#   module load cuda/11.8.0-gcc-7.2.0-xqzqlf2
#
# Then run from repo root:
#   bash setup_umn.sh
#
# Be sure to change the ENV_NAME and the MAMBA location to your need.
set -euo pipefail

MAMBA="$HOME/micromamba/micromamba"
ENV_NAME="mintenv"
PY_VER="3.11"

# init micromamba for this shell
eval "$($MAMBA shell hook --shell bash)"

echo ">>> Creating / activating env: ${ENV_NAME}"
if ! micromamba env list | grep -q "^${ENV_NAME}\b"; then
    micromamba create -y -n "${ENV_NAME}" -c conda-forge python="${PY_VER}" pip
fi
micromamba activate "${ENV_NAME}"

echo ">>> Micromamba packages (pinned to match working env as closely as possible)..."
micromamba install -y -c conda-forge \
  "numpy=2.0.1" \
  "matplotlib=3.10.6" \
  "scipy=1.16.2" \
  "omegaconf=2.3.0" \
  "hydra-core=1.3.2" \
  "parmed=4.3.0" \
  "wandb=0.22.1" \
  "tqdm=4.67.1" \
  "mdtraj=1.11.0" \
  "zstandard=0.25.0"

echo ">>> Pip packages (Torch + PyG stack + e3nn + lightning + others; pinned)..."
python -m pip install --upgrade pip build wheel setuptools

# Torch (CUDA 11.8 wheels)
pip install \
  "torch==2.5.0+cu118" \
  "torchvision==0.20.0+cu118" \
  "torchaudio==2.5.0+cu118" \
  --index-url https://download.pytorch.org/whl/cu118

# PyG wheels matching torch 2.5.0 + cu118
pip install \
  "pyg-lib==0.4.0+pt25cu118" \
  "torch-scatter==2.1.2+pt25cu118" \
  "torch-sparse==0.6.18+pt25cu118" \
  "torch-cluster==1.6.3+pt25cu118" \
  "torch-spline-conv==1.2.2+pt25cu118" \
  -f https://data.pyg.org/whl/torch-2.5.0+cu118.html

pip install "torch-geometric==2.6.1"
pip install "e3nn==0.5.7"
pip install "lightning==2.5.5" "pytorch-lightning==2.5.5" "torchmetrics==1.8.2"
pip install "GPUtil==1.4.0"
pip install "pymatgen==2025.6.14"
pip install "ase==3.26.0"

echo ">>> Build & install your project (dev mode)..."
python -m build --wheel
pip install -e .

echo ">>> Done. Activate later with:  micromamba activate ${ENV_NAME}"