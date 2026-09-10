#!/usr/bin/env bash
# Starts the FastAPI backend on http://localhost:8000
set -e
cd "$(dirname "$0")/backend"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt
if [ ! -f "data/OHRC/ohrc_demo.png" ]; then
  echo "No dataset found -- generating the bundled synthetic demo dataset..."
  python3 scripts/generate_demo_dataset.py
fi
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
