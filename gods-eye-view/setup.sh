#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install -r smartcity/requirements.txt
python3 smartcity/pipeline/smartcity_pipeline.py | head -3
python3 smartcity/notebooks/make_notebooks.py
PUPPETEER_SKIP_DOWNLOAD=1 npm ci --no-audit --no-fund
echo "Done. Run: npm run dev  ->  http://localhost:5173/smartcity/index.html"
