#!/usr/bin/env bash
# ==============================================================================
#  CIVICPULSE SIGHT — macOS 1-Click Deployment Launcher
# ==============================================================================

# Ensure working directory is the repository root
cd "$(dirname "$0")" || exit 1

echo "======================================================================"
echo "   CIVICPULSE SIGHT — Urban & Geospatial Intelligence Platform"
echo "======================================================================"
echo ""

# 1. Check if Node.js is installed
if ! command -v node >/dev/null 2>&1; then
    echo "[ERROR] Node.js is not found in your PATH."
    echo "Please install Node.js (v20+ recommended):"
    echo "  - Via Homebrew: brew install node"
    echo "  - Or download from: https://nodejs.org/"
    echo ""
    read -n 1 -s -r -p "Press any key to exit..."
    echo ""
    exit 1
fi

# 2. Check and install dependencies if missing
if [ ! -d "node_modules" ]; then
    echo "[SETUP] First-time launch detected. Installing dependencies..."
    echo "Running npm install..."
    npm install
    if [ $? -ne 0 ]; then
        echo "[ERROR] Dependency installation failed."
        read -n 1 -s -r -p "Press any key to exit..."
        echo ""
        exit 1
    fi
    echo "[SETUP] Dependencies successfully installed."
    echo ""
fi

# 3. Inform user of access links
echo "======================================================================"
echo "  CIVICPULSE SIGHT IS READY TO LAUNCH!"
echo "======================================================================"
echo ""
echo "  Primary 3D Globe Cockpit:"
echo "  --> http://localhost:4173/"
echo ""
echo "  Urban Priority & Infrastructure Engine:"
echo "  --> http://localhost:4173/smartcity/index.html"
echo ""
echo "  Triage Audit & Model Validation:"
echo "  --> http://localhost:4173/audit.html"
echo ""
echo "======================================================================"
echo "  Opening default browser in 3 seconds..."
echo "  Press Ctrl+C at any time in this window to stop the server."
echo "======================================================================"
echo ""

# 4. Open default browser after 3 seconds in the background
(
    sleep 3
    if command -v open >/dev/null 2>&1; then
        open "http://localhost:4173/"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "http://localhost:4173/"
    fi
) &

# 5. Start the Vite dev server
npm run dev
