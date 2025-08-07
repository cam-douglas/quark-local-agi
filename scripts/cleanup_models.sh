#!/usr/bin/env bash
#
# cleanup_models.sh
#   Move large model files from ~/ into ~/quark/models/

set -e

PROJECT_DIR="$HOME/quark"
MODELS_DIR="$PROJECT_DIR/models"

# 1. Make sure the target exists
mkdir -p "$MODELS_DIR"

# 2. Find and move any heavyweight files (over 50 MB) of these extensions:
find "$HOME" -maxdepth 1 -type f \
     \( -iname '*.bin' -o -iname '*.safetensors' -o -iname '*.json' \) \
     -size +50M \
     -print0 \
  | while IFS= read -r -d '' file; do
      echo "Moving $(basename "$file") → $MODELS_DIR/"
      mv "$file" "$MODELS_DIR/"
    done

echo "✅ All large model files moved to $MODELS_DIR"

