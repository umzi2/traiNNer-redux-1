#!/bin/bash

uv venv --python 3.12
source venv/bin/activate
uv pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match --no-cache --link-mode=copy
uv pip install . --index-strategy unsafe-best-match --link-mode=copy --no-cache

echo "traiNNer-redux dependencies installed successfully!"