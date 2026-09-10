#!/usr/bin/env bash
# Serves the static frontend on http://localhost:5173
set -e
cd "$(dirname "$0")/frontend"
python3 -m http.server 5173
