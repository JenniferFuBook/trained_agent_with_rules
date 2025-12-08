# ============================================
# Dockerfile for GPT-2 Training
# ============================================
# This file defines the container environment for training.
# It creates a clean, isolated environment with all dependencies.

# ============================================
# Base Image
# ============================================
# Use official Python 3.10 slim image (Debian-based, minimal size)
FROM python:3.10-slim

# ============================================
# Working Directory
# ============================================
# Set the working directory inside the container
WORKDIR /app

# ============================================
# System Dependencies
# ============================================
# Install build tools needed to compile Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# Note: We clean up apt cache to reduce image size

# ============================================
# Python Dependencies
# ============================================
# Copy requirements file first (for better Docker layer caching)
# If requirements.txt doesn't change, this layer is cached
COPY requirements.txt .

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir: Don't save pip cache (reduces image size)

# ============================================
# Copy Application Files
# ============================================
# Copy the training script into the container
COPY train.py .

# Copy training data (your text files)
COPY data ./data

# ============================================
# Environment Variables
# ============================================
# Set environment variables to prevent threading issues
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# ============================================
# Container Entry Point
# ============================================
# Command to run when container starts
CMD ["python", "train.py"]
