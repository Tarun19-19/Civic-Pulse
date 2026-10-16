# Windows one-shot setup:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Write-Host "1/4 Python deps"; python -m pip install -r smartcity/requirements.txt
Write-Host "2/4 Build data (CSV -> JSON)"; python smartcity/pipeline/smartcity_pipeline.py | Select-Object -First 3
Write-Host "3/4 Notebooks"; python smartcity/notebooks/make_notebooks.py
Write-Host "4/4 Node deps"; $env:PUPPETEER_SKIP_DOWNLOAD = "1"; npm ci --no-audit --no-fund
Write-Host "Done. Run: npm run dev  ->  http://localhost:5173/smartcity/index.html"
