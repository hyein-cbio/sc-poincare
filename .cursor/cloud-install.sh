#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --user --upgrade pip
python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --user -e '.[dev,benchmark]'
mkdir -p /home/ubuntu/.config/matplotlib
echo 'backend: Agg' > /home/ubuntu/.config/matplotlib/matplotlibrc
sudo ln -sfn /home/ubuntu/.local/bin/pytest /usr/local/bin/pytest
sudo ln -sfn /home/ubuntu/.local/bin/scanpy /usr/local/bin/scanpy
