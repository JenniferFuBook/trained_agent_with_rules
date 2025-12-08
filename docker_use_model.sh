#!/bin/bash
# ============================================
# Use Trained Model with Docker
# ============================================
# This avoids all macOS dependency issues!

echo "🐳 Running model in Docker container..."
echo ""

docker run --rm \
  -v "$(pwd)/agent-gpt2:/app/agent-gpt2" \
  -v "$(pwd)/use_model.py:/app/use_model.py" \
  python:3.10-slim \
  bash -c "
    pip install -q transformers torch tqdm && \
    python /app/use_model.py
  "

echo ""
echo "✅ Done!"
