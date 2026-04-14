#!/usr/bin/env bash
# Start backend + frontend for the Smart Review Gap-Filler demo
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Load .env
if [ -f "$ROOT/.env" ]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

echo "Starting FastAPI backend on :8000 …"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
# Prefer conda if available
if [ -x "/Users/sanjanatata/miniconda3/bin/python3" ]; then
  PYTHON="/Users/sanjanatata/miniconda3/bin/python3"
fi

$PYTHON -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "Starting React frontend on :5173 …"
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✦  ReviewIQ running at http://localhost:5173"
echo "   API docs at   http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
