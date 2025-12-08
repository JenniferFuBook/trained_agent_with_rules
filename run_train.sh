#!/bin/bash
# ============================================
# GPT-2 Training Launcher Script
# ============================================
# This script sets up environment variables before running the training script.
# Environment variables help prevent threading errors on macOS.
#
# Usage:
#   ./run_train.sh
#
# If you get "permission denied", run this first:
#   chmod +x run_train.sh

echo "🔧 Setting up environment for training..."

# ============================================
# Threading Configuration
# ============================================
# Limit threading to prevent mutex errors on macOS
# These settings force libraries to use single-threaded mode

export OMP_NUM_THREADS=1              # OpenMP threads (for parallel processing)
export MKL_NUM_THREADS=1              # Intel Math Kernel Library threads
export OPENBLAS_NUM_THREADS=1         # OpenBLAS library threads
export VECLIB_MAXIMUM_THREADS=1       # Apple's Accelerate framework threads
export NUMEXPR_NUM_THREADS=1          # NumExpr library threads

# ============================================
# Library-Specific Settings
# ============================================
export TOKENIZERS_PARALLELISM=false              # Disable tokenizer parallelism
export KMP_DUPLICATE_LIB_OK=TRUE                 # Allow duplicate library loads
export KMP_INIT_AT_FORK=FALSE                    # Don't initialize at fork
export PYTORCH_ENABLE_MPS_FALLBACK=1             # Enable Metal Performance Shaders fallback
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES   # Disable Objective-C fork safety (macOS)

echo "✅ Environment configured"
echo ""

# ============================================
# Run the Training Script
# ============================================
echo "🚀 Starting training script..."
echo ""

python3 train.py

# ============================================
# Check Exit Status
# ============================================
if [ $? -eq 0 ]; then
    echo ""
    echo "✨ Training completed successfully!"
else
    echo ""
    echo "❌ Training failed. Check the error messages above."
    exit 1
fi
