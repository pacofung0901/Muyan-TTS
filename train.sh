#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
LLAMA_DATA_DIR="$SCRIPT_DIR/llama-factory/data"
TRAINING_DIR="$SCRIPT_DIR/training"

# Run dataset preparation
echo "Preparing SFT dataset..."
python "$SCRIPT_DIR/prepare_sft_dataset.py"

# Copy data to llama-factory
echo "Copying dataset to llama-factory..."
cp "$LLAMA_DATA_DIR/dataset_info.json" "$DATA_DIR"

# Update dataset info
echo "Updating dataset info..."
python "$TRAINING_DIR/update_dataset_info.py"

# Prepare base model
echo "Preparing base model..."
python "$TRAINING_DIR/prepare_base_model.py"

# Start training
echo "Starting training..."
llamafactory-cli train "$TRAINING_DIR/sft.yaml"

echo "Training completed successfully!"